# code to calculate loss and pre-emphasis
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k


def loss_fn(y, y_pred):
    # error to signal loss function
    # y,y_pred of shape batch_size,sequence length,1

    y = tf.cast(y, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred_p = tf.concat((y_pred[:, :1, :], y_pred[:, 1:, :] - 0.85 * y_pred[:, :-1, :]), axis=1)
    # print(y_pred_p)
    y_p = tf.concat((y[:, :1, :], y[:, 1:, :] - 0.85 * y[:, :-1, :]), axis=1)
    # print(y_p)
    take_dc = True
    esr_loss = tf.reduce_sum(tf.square(y_pred_p - y_p), axis=1)
    # print('pre av esr', esr_loss)
    # print(tf.reduce_sum(tf.square(y_p), axis=1))
    esr_loss /= tf.reduce_sum(tf.square(y_p), axis=1)
    # print('post av esr', esr_loss)
    # n = y.shape[1]
    if take_dc:
        dc_loss = tf.square((tf.reduce_mean(y - y_pred, axis=1)))
        dc_loss /= tf.reduce_mean(tf.square(y), axis=1)
    else:
        dc_loss = 0
    # print(dc_loss)
    loss = esr_loss + dc_loss
    return loss


# y = np.ones((2,10,1))
# print(y[:,:1,:])

# print(loss_fn(y,y_pred))