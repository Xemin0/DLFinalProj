"""
GAN losses and Metrics
    
Inputs:
- logits_real: Tensor, shape [batch_size, 1], output of discriminator for each real image
- logits_fake: Tensor, shape[batch_size, 1], output of discriminator for each fake image
"""

bce_func = tf.keras.backend.binary_crossentropy
acc_func = tf.keras.metrics.binary_accuracy

'''
Vanilla GAN

# Loss functions!
def d_loss(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    return tf.math.reduce_mean(bce_func(tf.zeros_like(d_fake), d_fake) + bce_func(tf.ones_like(d_real),d_real))

def g_loss(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(bce_func(tf.ones_like(d_fake), d_fake))

# Accuracy functions!
def d_acc_fake(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    return acc_func(tf.zeros_like(d_fake), d_fake)

def d_acc_real(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    return acc_func(tf.ones_like(d_real), d_real)

def g_acc(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    return acc_func(tf.ones_like(d_fake), d_fake)
'''

'''
WGAN-GP
'''
# Loss Functions
# New Discriminator Loss WGAN
def d_wloss(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    real_loss = tf.reduce_mean(d_real)
    fake_loss = tf.reduce_mean(d_fake)
    return fake_loss - real_loss # as we want to minimize the incorrect guesses and maximize the correct guesses

# New Generator Loss WGAN
def g_wloss(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(d_fake)

'''
Will not be Used ??!!
'''
# Acc Functions (Discriminator/Critic Output Logits instead of Probabilities)
def d_wacc_fake(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    return acc_func(tf.zeros_like(d_fake), tf.nn.sigmoid(d_fake))

def d_wacc_real(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    return acc_func(tf.ones_like(d_real), tf.nn.sigmoid(d_real))

def g_wacc(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    return acc_func(tf.ones_like(d_fake), tf.nn.sigmoid(d_fake))
