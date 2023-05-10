from .GAN_Core import GAN_Core
class WGAN_Core(GAN_Core):
    '''
    WGAN Base Class
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rename Discriminator and its method to Critic and Criticize
        self.criticize = self.discriminate
        self.crt_model = self.dis_model

    def compile(self, optimizers, losses, **kwargs):
        super().compile(
            loss        = losses.values(),
            optimizer   = optimizers.values(),
#            metrics     = accuracies.values(),
            **kwargs
        )
        self.loss_funcs = losses
        self.optimizers = optimizers
#        self.acc_funcs  = accuracies


    def fit(self, *args, gp_weight=10, dis_steps=1, gen_steps=1, **kwargs):
        self.gen_steps = gen_steps
        self.dis_steps = dis_steps
        self.gp_weight = gp_weight                ## gradient penalty weight
        super().fit(*args, **kwargs)
