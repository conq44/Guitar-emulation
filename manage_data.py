import numpy as np


def get_dataset(x, y, sequence_length):

    """returns a tuple (x,y) each with shape ( num_samples, sequence_length, feature_size = 1)
    split into 0.5sec intervals which translates to a sequence length of 22050 @44.1 kHz"""

    num_samples = x.shape[0] // sequence_length
    x = np.reshape(x[:sequence_length * num_samples], (num_samples, sequence_length, 1))
    y = np.reshape(y[:sequence_length * num_samples], (num_samples, sequence_length, 1))
    # print(x)
    # batch_size = 3
    # num_batches = x.shape[0] // batch_size
    # x = np.reshape(x[:num_batches * batch_size, :], (batch_size, num_batches, sequence_length, 1))
    # y = np.reshape(y[:num_batches * batch_size, :], (batch_size, num_batches, sequence_length, 1))
    data = (x, y)
    return data


def shuffled(x, y):
    """ shuffles the training segments across the first two dimensions keeping segments continuous"""
    x = np.random.permutation(x)
    y = np.random.permutation(y)
    return x,y


# x = np.linspace(1, 10, 10)
# print(x)
# y = np.linspace(1, 10, 10)
# print(get_dataset(x,y,2)[0])

