import numpy as np
from NNMath import NNMath
import math
from Reader import DataReader
from random import shuffle
my_data, seq_lengths_train = DataReader.read_from_file('MousePred/Data/MouseData.txt')
my_data_validation, seq_lengths_val = DataReader.read_from_file('MousePred/Data/validationData.txt')
max_seq = np.max(seq_lengths_train + seq_lengths_val)
min_seq = np.min(seq_lengths_train + seq_lengths_val)


class NN(NNMath):
    def __init__(self, input_size, num_hidden_units, output_size, num_hidden_layers, min_val, max_val):
        NNMath.__init__(self)
        self.input_size = input_size
        self.num_hidden_units = num_hidden_units
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = []
        self.layer_nodes = []
        self.layer_nodes.append(np.zeros(input_size))
        self.layer_weights.append(self.init_weights(input_size, num_hidden_units))
        self.bayes = []
        self.bayes.append(np.zeros(num_hidden_units))
        self.max_val_out = max_val
        self.min_val_out = min_val
        self.min_val_in = 0
        self.max_val_in = 1000
        self.learning_rate = 0.1
        self.grad_eps = 0.000000001
        self.grad_check = True
        for i in range(num_hidden_layers-1):
            self.layer_weights.append(self.init_weights(num_hidden_units, num_hidden_units))
            self.layer_nodes.append(np.zeros(num_hidden_units))
            self.bayes.append(np.zeros(num_hidden_units))
        self.bayes.append(np.zeros(output_size))
        self.layer_nodes.append(np.zeros(num_hidden_units))
        self.layer_weights.append(self.init_weights(num_hidden_units, output_size))
        self.layer_nodes.append(np.zeros(output_size))

    def grad_checking(self, input, target, d_weights, d_bayeses):
        self.grad_check = True
        weight_array = np.array([])
        derivative_array = np.array([])
        grad_rez = np.array([])
        for i in range(self.num_hidden_layers+1):
            weight_array = np.append(weight_array, np.array(self.layer_weights[i]).reshape(-1, ))
        for i in range(self.num_hidden_layers+1):
            weight_array = np.append(weight_array, np.array(self.bayes[i]).reshape(-1, ))

        derivative_array = np.append(derivative_array, np.array(d_weights))
        derivative_array = np.append(derivative_array, np.array(d_bayeses))

        for i in range(len(weight_array)):
            weight_array[i] += self.grad_eps
            weight_acc, bayes_acc = self.array_to_matrices(weight_array)
            self.layer_weights = weight_acc
            self.bayes = bayes_acc
            trash, loss1 = self.train(input, target)
            weight_array[i] -= 2*self.grad_eps
            weight_acc, bayes_acc = self.array_to_matrices(weight_array)
            self.layer_weights = weight_acc
            self.bayes = bayes_acc
            trash, loss2 = self.train(input, target)

            grad_rez = np.append(grad_rez, ((loss1 - loss2)/(2*self.grad_eps)))
            weight_array[i] += self.grad_eps

        rez = np.linalg.norm((grad_rez-derivative_array))/(np.linalg.norm(grad_rez) + np.linalg.norm(derivative_array))
        self.grad_check = False
        return rez

    def reset_getes(self):
        self.layer_nodes = []
        self.layer_nodes.append(np.zeros(self.input_size))
        for i in range(self.num_hidden_layers):
            self.layer_nodes.append(np.zeros(self.num_hidden_units))
        self.layer_nodes.append(np.zeros(self.output_size))

    def array_to_matrices(self, arr_to_conv):
        weight_acc = []
        bayes_acc = []
        for i in range(self.num_hidden_layers+1):
            mat_size = self.layer_weights[i].size
            temp = arr_to_conv[:mat_size]
            weight_acc.append(np.array(temp).reshape(self.layer_weights[i].shape))
            arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)
        for i in range(self.num_hidden_layers+1):
            mat_size = self.bayes[i].size
            temp = arr_to_conv[:mat_size]
            bayes_acc.append(np.array(temp).reshape(self.bayes[i].shape))
            arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)

        return weight_acc, bayes_acc

    @staticmethod
    def init_weights(row_size, column_size):
        return 2 * np.random.random((row_size, column_size)) - 1

    def train(self, input, target):
        rez = self.feed_forward(input)
        loss = self.backward_prop(target, input)#self.scale_one(target, self.min_val_out, self.max_val_out), input)
        return rez, loss#math.ceil(self.scale_back(rez, self.min_val_out, self.max_val_out)), loss

    def feed_forward(self, input):
        self.layer_nodes[0] = np.array(input)
        for i in range(self.num_hidden_layers):
            self.layer_nodes[i+1] = self.sigmoid(np.dot(self.layer_nodes[i], self.layer_weights[i]) + self.bayes[i])
        self.layer_nodes[self.num_hidden_layers + 1] = self.sigmoid(np.dot(self.layer_nodes[self.num_hidden_layers], self.layer_weights[self.num_hidden_layers]) + self.bayes[self.num_hidden_layers])

        return self.layer_nodes[self.num_hidden_layers + 1]

    def backward_prop(self, target, input=None):
        Err = [None] * (self.num_hidden_layers + 2)
        loss = self.loss_multi(self.layer_nodes[self.num_hidden_layers + 1], target)
        Err[self.num_hidden_layers + 1] = self.sigmoid_derivative(self.layer_nodes[self.num_hidden_layers + 1]) * (target - self.layer_nodes[self.num_hidden_layers + 1])
        for t in reversed(range(1, self.num_hidden_layers+1)):
            Err[t] = np.dot(Err[t+1], self.layer_weights[t].T) * self.sigmoid_derivative(self.layer_nodes[t])

        #if not self.grad_check and input is not None:
        #    d_weights, d_bayeses = self.weight_derivatives(Err)
        #    self.grad_checking(input, target, d_weights, d_bayeses)

        self.weight_update(Err)
        self.reset_getes()
        return loss

    def weight_derivatives(self, Err):
        dWeight = np.array([])
        dBayes = np.array([])
        for t in range(self.num_hidden_layers+1):
            dWeight = np.append(dWeight,  np.dot(np.atleast_2d(self.layer_nodes[t]).T, np.atleast_2d(Err[t+1])).reshape(-1, ))
            dBayes = np.append(dBayes, Err[t+1].reshape(-1, ))
        return dWeight, dBayes

    def weight_update(self, Err):
        for t in range(self.num_hidden_layers+1):
            self.layer_weights[t] += self.learning_rate * np.dot(np.atleast_2d(self.layer_nodes[t]).T, np.atleast_2d(Err[t+1]))
            self.bayes[t] += self.learning_rate * Err[t+1]
            pass

    @staticmethod
    def scale_back(data, min_val, max_val):
        return (data * (max_val - min_val)) + min_val

    @staticmethod
    def scale_one(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)

'''
data_in = my_data[3]
last_point = data_in[len(data_in) - 1]
first_point = data_in[0]

first_last = np.hstack((np.array(first_point)[0:2], np.array(last_point)[0:2]))

Length_pred = NN(4, 20, 1, 2, min_seq, max_seq)
Length_pred.train(first_last, len(data_in))
'''

#print(Length_pred.scale_back(0.12, Length_pred.min_val_out, Length_pred.max_val_out))
Length_pred = NN(2, 5, 1, 2, min_seq, max_seq)
'''
for i in range(1000):
    loss = 0
    a = 0
    #shuffle(my_data)
    for data_in in my_data:
        if len(data_in) <= 2:
            continue
        last_point = data_in[len(data_in) - 1]
        first_point = data_in[0]

        first_last = np.hstack((np.array(first_point)[0:2], np.array(last_point)[0:2]))

        seq_length, l = Length_pred.train(first_last, len(data_in))

        loss += abs(l)
        #print(seq_length)
        if a == 0:
            #print(loss)
            pass
        a += 1

    print(loss/a)
'''
for i in range(500):
    loss = 0
    a = 0
    my_data = [[1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0]]
    shuffle(my_data)
    for data_in in my_data:
        if len(data_in) <= 2:
            continue
        last_point = data_in[len(data_in) - 1]

        seq_length, l = Length_pred.train(data_in[0:2], last_point)

        print(data_in[0:2], seq_length)
        loss += abs(l)
        #print(seq_length)
        if a == 0:
            #print(loss)
            pass
        a += 1

    #print(loss/a)

