import numpy as np
class NNMath:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def loss_multi(y, t):
        return np.sum(y - t)

    @staticmethod
    def loss_function(pred, target):
        return (np.linalg.norm(pred-target)**2)/len(pred)

    @staticmethod
    def loss(pred, label):
        return (pred[0][0] - label) ** 2

    @staticmethod
    def sigmoid_derivative(values):
        return values * (1 - values)

    @staticmethod
    def tanh(values):
        return np.tanh(values)

    @staticmethod
    def tanh_derivative(values):
        return 1 - values ** 2

    @staticmethod
    def tanh_derivative2(values):
        return 1 - (np.tanh(values)) ** 2

    @staticmethod
    def soft_max(values):
        return np.exp(values) / np.sum(np.exp(values))

    @staticmethod
    def scale_to_one(data):
        temp_min = data.min(axis=0)
        temp_max = data.max(axis=0)
        np.set_printoptions(threshold=5)
        temp = (data - temp_min) / (temp_max - temp_min)
        return [temp, temp_min, temp_max]

    @staticmethod
    def scale_from_one(data, min, max):
        temp = (data * (max - min)) + min
        return temp

    @staticmethod
    def scale_from_one(data, min_val, max_val):
        return (data * (max_val - min_val)) + min_val

    @staticmethod
    def scale_to_one(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)

