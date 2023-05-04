import tensorflow as tf
from tensorflow.math import log, exp


'''
Log-Space Sinkhorn Algorithm for Optimal Transport Plan

Provide Batch-Level Support
'''

def log_sum_exp(u: tf.Tensor, axis: int):
    # Reduce log sum exp along axis
    u_max = tf.reduce_max(u, axis = axis, keepdims = True)
    log_sum_exp_u = log( tf.reduce_sum(exp(u - u_max), axis = axis) ) + tf.reduce_sum(u_max, axis = axis)
    return log_sum_exp_u


def log_sinkhorn(M: tf.Tensor, reg: float, num_iters: int):
    '''
    Log-space-Sinkhorn Algorithm for Better Stability
    '''
    # Batch-Level Sinkhorn Iteration
    if M.ndim > 2:
        return batched_log_sinkhorn(M = M, reg = reg, num_iters = num_iters)

    # Initialize dual variable v (u is implicitly defined in the loop)
    log_v = tf.zeros(M.shape[1], dtype = tf.float32)


    # Exponentiate the pairwise distance matrix
    log_K = -tf.cast(M, dtype = tf.float32 )/ reg

    # Main Loop
    for i in range(num_iters):
        # Match r marginals
        log_u = -log_sum_exp(log_K + log_v[None, :], axis = 1)

        # Match c marginals
        log_v = -log_sum_exp(log_u[:, None] + log_K, axis = 0)

    # Compute optimal plan, cost, return everything
    log_P = log_u[:, None] + log_K + log_v[None, :]
    return log_P


def batched_log_sinkhorn(M, reg: float, num_iters: int):
    '''
    Batched Version of log-space-sinkhorn
    '''
    batch_size, x_points, _ = M.shape
    # both marginals are fixed with equal weights
    mu = tf.ones(shape = (batch_size, x_points), dtype = tf.float32) * (1.0/x_points)
    nu = tf.ones(shape = (batch_size, x_points), dtype = tf.float32) * (1.0/x_points)

    u = tf.zeros_like(mu)
    v = tf.zeros_like(nu)

    # To check if algorithm terminates because of threshold
    # or max iterations reached
    actual_nits = 0
    # Stopping Criterion
    thresh = 1e-1

    def C(M, u, v, reg):
        '''Modified Cost for Logarithmic Updates'''
        return (-M + tf.expand_dims(u, -1) + tf.expand_dims(v, -2) ) / reg

    # Sinkhorn iterations
    for i in range(num_iters):
        u1 = u # to check the update
        u = reg * ( log(mu + 1e-8) - tf.reduce_logsumexp(C(M, u, v, reg), axis = -1)) + u
        v = reg * ( log(nu + 1e-8) - tf.reduce_logsumexp(C(M, u, v, reg).transpose(-2,-1), axis = -1)) + v
        err = tf.mean( tf.reduce_mean(tf.abs(u - u1), axis = -1))

        actual_nits += 1
        if err.item() < thresh:
            break


    U, V = u, v
    # Transport plan pi = diag(a)*K*diag(b)
    log_p = C(M, U, V, reg)
    return log_p
