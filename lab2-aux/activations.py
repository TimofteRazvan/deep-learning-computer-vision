import numpy as np
#import SoftmaxClassifier as sc

def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    # TODO your code here
    exp_values = np.exp(x[:, np.newaxis] / t)
    softmax_values = exp_values / np.sum(exp_values, axis=0)
    # end TODO your code here
    return softmax_values

