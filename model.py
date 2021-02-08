import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wave
import scipy.io.wavfile as wavfile
from loss_fn import *


def get_dataset(x, y, sequence_length):
    """returns a tuple (x,y) each with shape ( num_samples, sequence_length, feature_size = 1)
    split into 0.5sec intervals which translates to a sequence length of 22050 @44.1 kHz"""
    num_samples = x.shape[0] // sequence_length
    x = np.reshape(x[:sequence_length * num_samples], (num_samples, sequence_length, 1))
    y = np.reshape(y[:sequence_length * num_samples], (num_samples, sequence_length, 1))
    data = (x, y)
    return data


def shuffled(x, y):
    """ shuffles the training segments """
    x = np.random.permutation(x)
    y = np.random.permutation(y)
    return x, y


def normalize(x):
    x = np.reshape(x, (x.shape[0], 1))
    # print('max', np.max(x))
    # print('norm', np.np.linalg.norm(x, ord=2))
    x = (np.max(x)-x)/(np.min(x)-x)
    return x

# def loss_fn(y, y_pred):
#     """ESR loss function with pre emphasis and dc loss component"""
#     # y,y_pred of shape batch_size,sequence length,1
#     # print(y)
#     y = tf.cast(y, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#     # pre emphasising with a first order high pass filter
#     y_pred_p = tf.concat([y_pred[:1], y_pred[1:] - 0.85 * y_pred[:-1]], axis=0)
#     y_p = tf.concat([y[:1], y[1:] - 0.85 * y[:-1]], axis=0)
#     # ESR loss
#     esr_loss = tf.reduce_sum(tf.square(y_pred_p - y_p), axis=0)
#     esr_loss /= tf.reduce_sum(tf.square(y_p), axis=0)
#     print('esr', esr_loss)
#     # DC loss
#     dc_loss = tf.square(tf.reduce_mean(y - y_pred, axis=0))
#     dc_loss /= tf.reduce_mean(tf.square(y), axis=0)
#     print('dc', dc_loss)
#     loss = esr_loss + dc_loss
#     # print(loss.shape)
#     return loss


# class MyModel(tf.keras.Model):
#     def __init__(self, batch_size):
#         super(MyModel, self).__init__()
#         xavier = tf.keras.initializers.GlorotUniform()
#         self.inp_layer = tf.keras.layers.Input(batch_shape=(batch_size, 22500, 1))
#         self.l1 = tf.keras.layers.LSTM(64, kernel_initializer=xavier, activation=tf.nn.relu, stateful=True)(self.inp_layer)
#         self.l2 = tf.keras.layers.Dense(1, kernel_initializer=xavier)(self.l1)
#         self.train_op = tf.keras.optimizers.Adam(learning_rate=5e-4)
#
#     def call(self, inputs):
#         x = self.inp_layer(inputs)
#         x = self.l1(x)
#         x = self.l2(x)
#         return x


# def train(model, train_data, num_epochs, batch_size):
#     # x, y have shapes (num_samples,22500,1)
#     for epoch in range(num_epochs):
#         # shuffle segments
#         x_train, y_train = shuffled(train_data[0], train_data[1])
#         # reshape into mini batches
#         num_batches = x_train.shape[0] // batch_size
#         x_train = np.reshape(x_train[:num_batches * batch_size, :], (num_batches, batch_size, sequence_length, 1))
#         y_train = np.reshape(y_train[:num_batches * batch_size, :], (num_batches, batch_size, sequence_length, 1))
#         # train_data = (x_train, y_train)
#         for step in range(x_train.shape[0]):
#             x_batch_train = x_train[step]
#             y_batch_train = y_train[step]
#             model.reset_states()  # resetting the recurrent unit to zero state manually
#             # for the first 1000 samples only forward pass to initialise recurrent state
#             # print(x_batch_train.shape)
#             x_first_thou = x_batch_train[:, :1000, :]
#             init_out = model.predict(x_first_thou)
#             print(init_out.shape)
#             # split into sequences with size (batch_size, 2105, 1)
#             x_sequences_train = np.array(np.split(x_batch_train[:, 1000:, :], 10, axis=1))
#             y_sequences_train = np.array(np.split(y_batch_train[:, 1000:, :], 10, axis=1))
#             for sub_seq in range(x_sequences_train.shape[0]):
#                 # update params every 2105 samples without resetting state
#                 model.fit(x_sequences_train[sub_seq], y_sequences_train[sub_seq], batch_size=batch_size, verbose=1)


def train_custom(model, train_data, num_epochs, batch_size):
    # x, y have shapes (num_samples,22500,1)
    for epoch in range(num_epochs):
        # shuffle segments
        x_train, y_train = shuffled(train_data[0], train_data[1])
        # reshape into mini batches
        num_batches = x_train.shape[0] // batch_size
        x_train = np.reshape(x_train[:num_batches * batch_size, :], (num_batches, batch_size, sequence_length, 1))
        y_train = np.reshape(y_train[:num_batches * batch_size, :], (num_batches, batch_size, sequence_length, 1))
        # train_data = (x_train, y_train)
        for batch in range(num_batches):
            x_batch_train = x_train[batch]
            y_batch_train = y_train[batch]
            # for the first 1000 samples only forward pass to initialise recurrent state
            # print(x_batch_train.shape)
            x_first_thou = x_batch_train[:, :1000, :]
            print('x_first_thou', x_first_thou.shape)
            init_out = model(x_first_thou)
            print('init_out has shape', init_out)
            # split into sequences with size (batch_size, 2105, 1)
            x_sequences_train = np.array(np.split(x_batch_train[:, 1000:, :], 10, axis=1))
            y_sequences_train = np.array(np.split(y_batch_train[:, 1000:, :], 10, axis=1))
            for sub_seq in range(x_sequences_train.shape[0]):
                # update params every 2105 samples without resetting state
                # print(loss_fn(x_sequences_train[sub_seq],y_sequences_train[sub_seq]))
                # model.fit(x_sequences_train[sub_seq], y_sequences_train[sub_seq], batch_size=batch_size, verbose=1)
                # with tf.GradientTape() as tape:
                print('input', x_sequences_train[sub_seq])
                out = model.layers[1](x_sequences_train[sub_seq])
                out2 = model.layers[2](out)
                print('output_1', out)
                print('output_2', out2)
                loss = loss_fn(out2, y_sequences_train[sub_seq])
                print('loss', loss)

# _, x = wavfile.read(r'C:\Users\adit\PycharmProjects\Guitar-emulation\data\clean\clean_full_4.wav')
# _, y = wavfile.read(r'C:\Users\adit\PycharmProjects\Guitar-emulation\data\distorted\distorted_4_ac30.wav')
# print(x.shape)
# print(y.shape)
# sequence_length = 22050
# batch_size = 32
# x_f, y_f = get_dataset(x, y, sequence_length)
# print(x_f.shape)
# print(y_f.shape)

# xavier = tf.keras.initializers.GlorotUniform()
# inp_layer = tf.keras.layers.Input(batch_shape=(batch_size, None, 1))
# lstm_layer = tf.keras.layers.LSTM(64, kernel_initializer=xavier, activation=tf.nn.relu, stateful=True)(inp_layer)
# dense_layer = tf.keras.layers.Dense(1, kernel_initializer=xavier)(lstm_layer)
# train_op = tf.keras.optimizers.Adam(learning_rate=5e-4)
# AmpModel = tf.keras.Model(inp_layer, dense_layer)
# print(AmpModel.summary())
# AmpModel.compile(optimizer=train_op, loss=loss_fn)
# # print(AmpModel.summary())
# x_f =x_f.astype('float32')
# y_f =y_f.astype('float32')
# train(AmpModel, (x_f, y_f), 1, batch_size)


_, x = wavfile.read(r'C:\Users\adit\PycharmProjects\Guitar-emulation\data\clean\clean_full_4.wav')
_, y = wavfile.read(r'C:\Users\adit\PycharmProjects\Guitar-emulation\data\distorted\distorted_4_ac30.wav')
print('x', x.shape)
print('y', y.shape)
# plt.plot(y)
# plt.plot(x)
# plt.show()
sequence_length = 22050
batch_size = 32
x_f = normalize(x)
y_f = normalize(y)
x_f, y_f = get_dataset(x_f, y_f, sequence_length)
print('x_full', x_f.shape)
print('y_full', y_f.shape)
x_f = x_f.astype('float32')
y_f = y_f.astype('float32')
train_data = (x_f, y_f)
xavier = tf.keras.initializers.GlorotUniform()
inputs = tf.keras.Input(shape=(None,1), batch_size=32)
X = tf.keras.layers.LSTM(64, kernel_initializer=xavier, activation=tf.nn.relu, return_sequences=True)(inputs)
dense = tf.keras.layers.Dense(1)(X)
# print(dense.shape)
train_op = tf.keras.optimizers.Adam(learning_rate=5e-4)
AmpModel = tf.keras.Model(inputs=inputs, outputs=dense)
AmpModel.compile(optimizer=train_op, loss=loss_fn)
AmpModel.summary()
print('weights', AmpModel.layers[2].get_weights()[0])
train_custom(AmpModel, train_data, 1, batch_size)
