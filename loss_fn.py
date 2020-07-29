# code to calculate loss and pre-emphasis
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k

def loss_fn(y, y_pred):
    # error to signal loss function
    #y,y_pred of shape batch_size,sequence length,1

    y = tf.cast(y, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)  

    y_pred_p = tf.concat((tf.expand_dims(y_pred[:,0,:],axis =1), y_pred[:,1:,:] - 0.85 * y_pred[:,:-1,:]), axis = 1)
    y_p = tf.concat((tf.expand_dims(y[:,0,:],axis =1), y[:,1:,:] - 0.85 * y[:,:-1,:]), axis = 1)
 
    take_dc=True
    esr_loss = tf.reduce_sum(tf.square(y_pred_p - y_p), axis = 1)
    esr_loss /= tf.reduce_sum(tf.square(y_p), axis = 1)
  
    n = y.shape[1]
    if take_dc:
        dc_loss = tf.square((1 / n) * tf.reduce_sum(y - y_pred, axis = 1))
        dc_loss /= (1 / n) * tf.reduce_sum(tf.square(y), axis = 1)
    else:
        dc_loss = 0
 
    loss = esr_loss + dc_loss
    return loss

# if __name__== '__main__':
#     y = [0,1,2,3,4,5]
#     y_pred = [1,2,3,4,5,6]
#     y = np.asarray(y)*1.0
#     y_pred = np.asarray(y_pred)*1.0
#     y_p = pre_emphasis(y, 0.85)
#     y_pred_p = pre_emphasis(y_pred, 0.85)
#     print(y_p, y_pred_p)
#     loss = loss_fn(y_pred,y,y_pred_p,y_p)

#     sess = tf.Session()
#     print(sess.run(loss))
#     sess.close()
