import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from .WGAN_Core import WGAN_Core
from model.metrics import reconstructionLoss

class SRWGAN(WGAN_Core):
    '''
    WGAN Class with Gradient Penalty and ContentLoss(a pretrained model)

    Generator:
        - input  SuperResolution Images
                 shape   (None, 64, 64,3) by default
                 range   (-1,1)
        - output shape   (None,256,256,3) by default
                 range   (-1,1)

        - Loss:
            - -D(G(Low_Res))
            - Reconstruction Loss
            - Content Loss (Weighted)

    Discriminator:
        - input  Super/High Resolution Images
                 shape   (None, 256, 256, 3) by default
                 range   (-1,1)
        - output shape   (None, 1)
                 range   (-inf, +inf)

        - Loss:
            - Wasserstein Distance (D(fake) - D(real))
            - Gradient-Penalty (Weighted)

    *** For Content Loss, by default we use a pretrained ResNet50 and
       take intermediate features at layer 2, 7, 10, 14
    '''
    def __init__(self, pretrained = 'resnet50', hyperimg_ids = [2,7,10,14],\
                 lr_shape = [64,64,3], hr_shape = [256,256,3],\
                 name = 'srwgan', *args, **kwargs):
        super().__init__(name = name, *args, **kwargs)
        '''
        Pretrained Models for Content Loss.
        Currently Support:
                    - ResNet50
                    - ResNet101

                    - VGG
                    - FCN101
        '''
        ## ** Hard Coded For now ** ##
        ### Needs Updates ### 
        self.pretrained_name = pretrained
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape

        # Pretrained Model for the Content Loss
        self.pretrained = self.build_pretrained(pretrained, hyperimg_ids, hr_shape)
        self.pretrained.trainable = False

        self.hyper_ids = hyperimg_ids

    '''
    Content Loss Part
    '''
    @staticmethod
    def build_pretrained(pretrained, hyperimg_ids, input_shape):
        if 'resnet50' == pretrained.lower():
            model = tf.keras.applications.ResNet50(input_shape = input_shape, include_top = False)
            # hyperimg_ids =  [0,11,12,13] for CAM
        elif 'resnet101' == pretrained.lower():
            model = tf.keras.applications.ResNet101(input_shape = input_shape, include_top = False)
            # hyperimg_ids = [2,22,24,25,27,28,29] for CAM
            # hyperimg_ids = [0,19,27,28,29,30]
        else:
            raise Exception('So far only supports resnet50 and resnet 101 for ContentLoss')

        outputs = [model.layers[i].output for i in hyperimg_ids]

        # Create a new Model that outputs the intermediate features from the pretrained model
        # Input shape (batch_sz, H, W, C)
        # Output Shape (batch_sz, fh, fw, c) for each layer 
        return Model(inputs = model.input, outputs = outputs)

    def contentLoss(self, img_fake:tf.Tensor, img_real:tf.Tensor):
        '''
        # MSE of Hyperimages(SuperResolutionImage - HighResolutionImage) where SRImages are generated
        # Hyper Images are constructed by taking intermediate output feature maps
        # from a Pretrained model (i.g.ResNet50)
        '''

        # Forward Pass of the Pretrained Model
        # which returns intermediate output feature maps at specified hyperimg_ids
        fakeFeatures = self.pretrained(img_fake)
        realFeatures = self.pretrained(img_real)

        ###******###
        total_loss = tf.reduce_sum([reconstructionLoss(fake, real) for fake, real in zip(fakeFeatures, realFeatures)])
        # record the total number of 'pixels'
        #total_num = tf.reduce_sum([tf.size(fake) for fake in fakeFeatures])

        # total_loss is the sum of MSE loss of each feature map
        # total_num should then be the num of feature maps
        total_num = tf.cast(len(self.hyper_ids), dtype = tf.float32)
        return total_loss / total_num

    '''
    Gradient Penalty Part
    '''
    def gradient_penalty(self, batch_size, x_fake, x_real):
        '''
        Calculate the Gradient Penalty.

        This loss is calculated on an interpolated image and added to the Discriminator Loss.
        '''

        # Get the interpolated image
        e_norm = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0) # Normal Distribution? Uniform Distribution?
        diff = x_fake - x_real
        interpolates = x_real + e_norm * diff

        # Gradients
        with tf.GradientTape() as tape:
          tape.watch(interpolates)
          # 1. Output of Critic using the interpolates
          pred = self.crt_model(interpolates, training = True)

        # Graidents w.r.t. the input = interpolates
        grads = tape.gradient(pred, [interpolates])[0]

        # Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis = [1,2,3])) # All dimensions except for the batch dimension
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)

        return gradient_penalty


    '''
    Test Subroutine for SRWGAN-GP
    '''
    def test_step(self, data):
        # Low Resolution, High Resolution
        lres, hres = data
        batch_size = tf.shape(lres)[0]

        ## - lres: Low Resolution Images from dataset (generator's input)

        ## - hres: High Resolution Images from dataset (ground truth; real)
        ## - d_real: The discriminator's prediction of the reals (High Resolution)
        ## - sres: Super Resolution Images generated by Generator (generated; fake)
        ## - d_fake: The discriminator's prediction of the fakes (Super Resolution)
        sres = self.generate(lres, training = False)
        d_fake = self.criticize(sres, training = False)
        d_real = self.criticize(hres, training = False)

        ###################################

        metrics = dict()

        '''
        all_funcs = {**self.loss_funcs, **self.acc_funcs}

        for key, func in all_funcs.items():
          metrics[key] = func(d_fake, d_real, x_fake, x_real)
        '''
        for key, func in self.loss_funcs.items():
            if 'd_loss' == key:
                metrics[key] = func(d_fake, d_real, None, None)
                metrics['gradient_penaty'] = self.gradient_penalty(batch_size, sres, hres)
                metrics['gp_weight'] = self.gp_weight
            elif 'g_loss' == key:
                metrics['content_weight'] = self.content_weight
                metrics['content_loss'] = self.contentLoss(sres, hres)
                metrics[key] = func(d_fake, None, sres, hres)
                metrics['g_loss_total'] = metrics[key] + self.content_weight * metrics['content_loss']
            else:
                raise Exception('d_loss and g_loss are recommended for Keys of Losses Dictionary.')
        return metrics

    '''
    Train Subroutine for SRWGAN-GP
    '''
    def train_step(self, data):
        lres, hres = data
        batch_size = tf.shape(lres)[0]

        # For each Batch, as laid out in the original paper:
        # 1. Train the Discriminator and Get the Discriminator Loss: (Wasserstein Distance + Gradient Penalty)
        # 2. Train the Generator and Get the generator Loss: (Feedback Criticism + Reconstruction Loss + (weighted)Content Loss + (OT?))
        # 3. Calculate the Gradient Penalty
        # 4. Multiply this gradient Penalty with a constant weight
        # 5. Add the Gradient Penalty to the discriminator Loss
        # 6. Return the Generator and Discriminator Losses as a loss dictionary

        # Train the Discriminator First.
        ##################################################

        # Train for Discriminator/Critic
        loss_fn = self.loss_funcs['d_loss']
        optimizer = self.optimizers['d_opt']

        for i in range(self.dis_steps):
          # Gradient Tape
          with tf.GradientTape() as tape:
            # Generated Fake images from the Z_samp
            sres = self.generate(lres, training = True) # True for Gradient Penalty
            # Logits/Criticism from Discriminator for the Fake images
            d_fake = self.criticize(sres, training = True)
            # Logits/Criticism from Discriminator for the Real images
            d_real = self.criticize(hres, training = True)

            # Default Discriminator Loss 
            d_cost = loss_fn(d_fake, d_real, None, None)
            # Gradient Penalty
            gp = self.gradient_penalty(batch_size, sres, hres)
            # Total loss = Default D Loss + Gradient-Penalty
            d_loss = d_cost + gp * self.gp_weight

          # Get the Gradients of d_loss w.r.t. the Discriminator's parameters
          g = tape.gradient(d_loss, self.crt_model.trainable_variables)
          optimizer.apply_gradients(zip(g, self.crt_model.trainable_variables))


        # Train for Generator
        loss_fn = self.loss_funcs['g_loss']
        optimizer = self.optimizers['g_opt']

        for i in range(self.gen_steps):
          # Gradient Tape
          with tf.GradientTape() as tape:
            # Generated Fake images from the z_samp
            sres = self.generate(lres, training = True)
            # Logits/Criticism from Discriminator for the fake images
            d_fake = self.criticize(sres, training = True)
            # Generator Loss
            ##  - -D(fake) + reconstructionLoss + contentLoss
            g_loss = loss_fn(d_fake, None, sres, hres) + self.content_weight * self.contentLoss(sres, hres)

          # Get the gradients of g_loss w.r.t. the Generator's parameters
          g = tape.gradient(g_loss, self.gen_model.trainable_variables)
          optimizer.apply_gradients(zip(g, self.gen_model.trainable_variables))

        # Compute Final states for metric computation
        sres = self.generate(lres, training = False)

        d_fake = self.discriminate(sres, training = False)

        d_real = self.discriminate(hres, training = False)

        #######################################

        metrics = dict()

        '''
        all_funcs = {**self.loss_funcs, **self.acc_funcs}

        for key, func in all_funcs.items():
            metrics[key] = func(d_fake, d_real)
        '''

        for key, func in self.loss_funcs.items():
            if 'd_loss' == key:
                metrics[key] = func(d_fake, d_real, None, None)
                metrics['gradient_penaty'] = self.gradient_penalty(batch_size, sres, hres)
                metrics['gp_weight'] = self.gp_weight
            elif 'g_loss' == key:
                metrics['content_weight'] = self.content_weight
                metrics['content_loss'] = self.contentLoss(sres, hres)
                metrics[key] = func(d_fake, None, sres, hres)
                metrics['g_loss_total'] = metrics[key] + self.content_weight * metrics['content_loss']
            else:
                raise Exception('d_loss and g_loss are recommended for Keys of Losses Dictionary.')

        return metrics

    # Add content_weight for ContentLoss in SRWGAN-GP
    def fit(self, *args, content_weight = 1e-3, **kwargs):
        self.content_weight = content_weight
        super().fit(*args, **kwargs)

    '''
    Save the Model
    '''

