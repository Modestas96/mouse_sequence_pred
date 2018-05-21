# Teaching a 3 level neural network to work as Full Adder

# import numpy for maths, pandas for reading data from text
import numpy as np
import csv
from os import getcwd, path, makedirs
from Reader import DataReader
from random import shuffle

# Neural Network class definition
class NeuralNetwork():
    def __init__(self, min_seq, max_seq, folder):
        # seed the random generator for easier debugging
        np.random.seed(1)

        # Save all variables in self for future references
        self.input_shape = (1, 1)
        self.output_shape = (1, 1)
        self.layer_1_nodes = 50
        self.layer_2_nodes = 50
        self.layer_3_nodes = 50
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.main_dir = getcwd() + '/SequenceLenPred/Models/'
        if not path.exists(self.main_dir + folder + '/'):
            makedirs(self.main_dir + folder + '/')
        self.main_dir = self.main_dir + folder + '/'
        # Generate weights with value between -1 to 1 so that mean is overall 0
        self.weights_1 = np.random.randn(self.input_shape[1], self.layer_1_nodes) * np.sqrt(1/self.input_shape[1])
        self.weights_2 = np.random.randn(self.layer_1_nodes, self.layer_2_nodes) * np.sqrt(1/self.layer_1_nodes)
        self.weights_3 = np.random.randn(self.layer_2_nodes, self.layer_3_nodes) * np.sqrt(1/self.layer_2_nodes)
        self.vw1 = np.zeros((self.input_shape[1], self.layer_1_nodes))
        self.vw2 = np.zeros((self.layer_1_nodes, self.layer_2_nodes))
        self.vw3 = np.zeros((self.layer_2_nodes, self.layer_3_nodes))
        self.vwo = np.zeros((self.layer_3_nodes, self.output_shape[1]))
        self.b1 = np.zeros((1, self.layer_1_nodes))
        self.b2 = np.zeros((1, self.layer_2_nodes))
        self.b3 = np.zeros((1, self.layer_3_nodes))
        self.bo = np.zeros((1, self.output_shape[1]))
        self.vb1 = np.zeros((1, self.layer_1_nodes))
        self.vb2 = np.zeros((1, self.layer_2_nodes))
        self.vb3 = np.zeros((1, self.layer_3_nodes))
        self.vbo = np.zeros((1, self.output_shape[1]))
        self.beta = 0.9
        self.learning_rate = 0.1
        self.out_weights = np.random.randn(self.layer_3_nodes, self.output_shape[1]) * np.sqrt(1/self.layer_3_nodes)
        self.grad_eps = 0.000000001
        self.grad_check = False

        self.do_grad = False
        self.grad_is_doing = False

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Reversed Sigmoid by derivating the value
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(x, 0, x)

    # Reversed Sigmoid by derivating the value
    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def think(self, x):
        # Multiply the input with weights and find its sigmoid activation for all layers
        layer1 = self.relu(np.dot(x, self.weights_1))
        layer2 = self.relu(np.dot(layer1, self.weights_2))
        layer3 = self.relu(np.dot(layer2, self.weights_3))
        output = self.sigmoid(np.dot(layer3, self.out_weights))
        return self.scale_back(output, self.min_seq, self.max_seq)

    def train(self, gateInput, gateOutput):
        gateInput = np.atleast_2d(gateInput)
        gateOutput = np.atleast_2d(self.scale_one(gateOutput, self.min_seq, self.max_seq))

        # Same as code of thinking
        layer1 = self.relu(np.dot(gateInput, self.weights_1) + self.b1)
        layer2 = self.relu(np.dot(layer1, self.weights_2) + self.b2)
        layer3 = self.relu(np.dot(layer2, self.weights_3) + self.b3)
        output = self.sigmoid(np.dot(layer3, self.out_weights) + self.bo)
        #print(gateInput, output)
        # What is the error?
        outputError = output - gateOutput
        # Find delta, i.e. Product of Error and derivative of next layer
        delta4 = outputError * self.sigmoid_derivative(output)

        # Multiply with transpose of last layer
        # to invert the multiplication we did to get layer

        # Procedure stays same, but the error now is the product of current weight and
        # Delta in next layer
        delta3 = np.dot(delta4, self.out_weights.T) * self.relu_derivative(layer3)
        delta2 = np.dot(delta3, self.weights_3.T) * self.relu_derivative(layer2)
        delta1 = np.dot(delta3, self.weights_2.T) * self.relu_derivative(layer1)

        out_weights_adjustment = np.dot(layer3.T, delta4)
        self.vwo = self.beta * self.vwo + (1-self.beta) * out_weights_adjustment
        self.vbo = self.beta * self.vbo + (1-self.beta) * delta4

        weight_3_adjustment = np.dot(layer2.T, delta3)
        self.vw3 = self.beta * self.vw3 + (1 - self.beta) * weight_3_adjustment
        self.vb3 = self.beta * self.vb3 + (1 - self.beta) * delta3

        weight_2_adjustment = np.dot(layer1.T, delta2)
        self.vw2 = self.beta * self.vw2 + (1 - self.beta) * weight_2_adjustment
        self.vb2 = self.beta * self.vb2 + (1 - self.beta) * delta2

        weight_1_adjustment = np.dot(gateInput.T, delta1)
        self.vw1 = self.beta * self.vw1 + (1 - self.beta) * weight_1_adjustment
        self.vb1 = self.beta * self.vb1 + (1 - self.beta) * delta1

        if not self.grad_is_doing:
            self.out_weights -= self.learning_rate * self.vwo
            self.bo -= self.learning_rate * self.vbo
            self.weights_3 -= self.learning_rate*self.vw3
            self.b3 -= self.learning_rate*self.vb3
            self.weights_2 -= self.learning_rate*self.vw2
            self.b2 -= self.learning_rate * self.vb2
            self.weights_1 -= self.learning_rate * self.vw1
            self.b1 -= self.learning_rate*self.vb1

        if not self.grad_is_doing and self.grad_check:
            d_weights = np.array([])
            d_bayeses = np.array([])
            weights = np.array([])
            bayeses = np.array([])
            d_weights = np.append(d_weights, np.atleast_2d(weight_1_adjustment).reshape(-1, ))
            d_weights = np.append(d_weights, np.atleast_2d(weight_2_adjustment).reshape(-1, ))
            d_weights = np.append(d_weights, np.atleast_2d(weight_3_adjustment).reshape(-1, ))
            d_weights = np.append(d_weights, np.atleast_2d(out_weights_adjustment).reshape(-1, ))
            d_bayeses = np.append(d_bayeses, np.atleast_2d(delta4).reshape(-1, ))
            d_bayeses = np.append(d_bayeses, np.atleast_2d(delta3).reshape(-1, ))
            d_bayeses = np.append(d_bayeses, np.atleast_2d(delta2).reshape(-1, ))
            d_bayeses = np.append(d_bayeses, np.atleast_2d(delta1).reshape(-1, ))

            weights = np.append(weights, np.atleast_2d(self.weights_1).reshape(-1, ))
            weights = np.append(weights, np.atleast_2d(self.weights_2).reshape(-1, ))
            weights = np.append(weights, np.atleast_2d(self.weights_3).reshape(-1, ))
            weights = np.append(weights, np.atleast_2d(self.out_weights).reshape(-1, ))

            bayeses = np.append(bayeses, np.atleast_2d(self.b3).reshape(-1, ))
            bayeses = np.append(bayeses, np.atleast_2d(self.b2).reshape(-1, ))
            bayeses = np.append(bayeses, np.atleast_2d(self.b1).reshape(-1, ))
            bayeses = np.append(bayeses, np.atleast_2d(self.bo).reshape(-1, ))

            self.grad_checking(gateInput, gateOutput,weights, bayeses, d_weights, d_bayeses)

        return outputError, self.scale_back(output, self.min_seq, self.max_seq)



    def grad_checking(self, inputt, targett, bayeses, weights, d_weights, d_bayeses):
        self.grad_is_doing = True
        weight_array = np.array([])
        derivative_array = np.array([])
        grad_rez = np.array([])
        weight_array = np.append(weight_array, np.array(weights))
        weight_array = np.append(weight_array, np.array(bayeses))

        derivative_array = np.append(derivative_array, np.array(d_weights))
        derivative_array = np.append(derivative_array, np.array(d_bayeses))

        for i in range(len(weight_array)):
            if i == len(weight_array) - 1:
                print('uess')
                pass
            weight_array[i] += self.grad_eps
            weight_acc, bayes_acc = self.array_to_matrices(weight_array)
            self.out_weights = weight_acc[3]
            self.weights_3 = weight_acc[2]
            self.weights_2 = weight_acc[1]
            self.weights_1 = weight_acc[0]
            self.bo = bayes_acc[3]
            self.b3 = bayes_acc[2]
            self.b2 = bayes_acc[1]
            self.b1 = bayes_acc[0]
            loss1, trash = self.train(inputt, targett)
            weight_array[i] -= 2*self.grad_eps
            weight_acc, bayes_acc = self.array_to_matrices(weight_array)
            self.out_weights = weight_acc[3]
            self.weights_3 = weight_acc[2]
            self.weights_2 = weight_acc[1]
            self.weights_1 = weight_acc[0]
            self.bo = bayes_acc[3]
            self.b3 = bayes_acc[2]
            self.b2 = bayes_acc[1]
            self.b1 = bayes_acc[0]
            loss2, trash = self.train(inputt, targett)

            if i == len(weight_array) - 1:
                grad_rez = np.append(grad_rez, derivative_array[i])
            else:
                grad_rez = np.append(grad_rez, ((loss1 - loss2) / (2 * self.grad_eps)))

            weight_array[i] += self.grad_eps


        rez = np.linalg.norm((grad_rez-derivative_array))/(np.linalg.norm(grad_rez) + np.linalg.norm(derivative_array))
        self.grad_is_doing = False
        return rez

    def array_to_matrices(self, arr_to_conv):
        weight_acc = []
        bayes_acc = []

        mat_size = self.input_shape[1] * self.layer_1_nodes
        shape = (self.input_shape[1], self.layer_1_nodes)
        temp = arr_to_conv[:mat_size]
        weight_acc.append(np.array(temp).reshape(shape))
        arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)

        mat_size = self.layer_1_nodes * self.layer_2_nodes
        shape = (self.layer_1_nodes, self.layer_2_nodes)
        temp = arr_to_conv[:mat_size]
        weight_acc.append(np.array(temp).reshape(shape))
        arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)

        mat_size = self.layer_2_nodes * self.layer_3_nodes
        shape = (self.layer_2_nodes, self.layer_3_nodes)
        temp = arr_to_conv[:mat_size]
        weight_acc.append(np.array(temp).reshape(shape))
        arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)

        mat_size = self.layer_3_nodes * 1
        shape = (self.layer_3_nodes, 1)
        temp = arr_to_conv[:mat_size]
        weight_acc.append(np.array(temp).reshape(shape))
        arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)

        mat_size = self.layer_1_nodes
        shape = (1, self.layer_1_nodes)
        temp = arr_to_conv[:mat_size]
        bayes_acc.append(np.array(temp).reshape(shape))
        arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)

        mat_size = self.layer_2_nodes
        shape = (1, self.layer_2_nodes)
        temp = arr_to_conv[:mat_size]
        bayes_acc.append(np.array(temp).reshape(shape))
        arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)

        mat_size = self.layer_3_nodes
        shape = (1, self.layer_3_nodes)
        temp = arr_to_conv[:mat_size]
        bayes_acc.append(np.array(temp).reshape(shape))
        arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)

        mat_size = self.output_shape[1]
        shape = (1, self.output_shape[1])
        temp = arr_to_conv[:mat_size]
        bayes_acc.append(np.array(temp).reshape(shape))
        arr_to_conv = np.delete(arr_to_conv, np.s_[:mat_size], None)

        return weight_acc, bayes_acc


    def export_weights(self):
        np.savetxt(self.main_dir + "wout.csv", self.out_weights, delimiter=",")
        np.savetxt(self.main_dir + "w3.csv", self.weights_3, delimiter=",")
        np.savetxt(self.main_dir + "w2.csv", self.weights_2, delimiter=",")
        np.savetxt(self.main_dir + "w1.csv", self.weights_1, delimiter=",")

        np.savetxt(self.main_dir + "bout.csv", self.bo, delimiter=",")
        np.savetxt(self.main_dir + "b3.csv", self.b3, delimiter=",")
        np.savetxt(self.main_dir + "b2.csv", self.b2, delimiter=",")
        np.savetxt(self.main_dir + "b1.csv", self.b1, delimiter=",")

    def import_weights(self):
        self.out_weights = np.array(list(csv.reader(open(self.main_dir + "wout.csv", "r"), delimiter=","))).astype(float)
        self.weights_3 = np.array(list(csv.reader(open(self.main_dir + "w3.csv", "r"), delimiter=","))).astype(float)
        self.weights_2 = np.array(list(csv.reader(open(self.main_dir + "w2.csv", "r"), delimiter=","))).astype(float)
        self.weights_1 = np.array(list(csv.reader(open(self.main_dir + "w1.csv", "r"), delimiter=","))).astype(float)

        self.bo = np.array(list(csv.reader(open(self.main_dir + "bout.csv", "r"), delimiter=","))).astype(float).T[0]
        self.b3 = np.array(list(csv.reader(open(self.main_dir + "b3.csv", "r"), delimiter=","))).astype(float).T[0]
        self.b2 = np.array(list(csv.reader(open(self.main_dir + "b2.csv", "r"), delimiter=","))).astype(float).T[0]
        self.b1 = np.array(list(csv.reader(open(self.main_dir + "b1.csv", "r"), delimiter=","))).astype(float).T[0]



    @staticmethod
    def scale_back(data, min_val, max_val):
        return (data * (max_val - min_val)) + min_val

    @staticmethod
    def scale_one(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)


if __name__ == '__main__':
    my_data, seq_lengths_train = DataReader.read_from_file('MousePred/Data/MouseData.txt')
    my_data_validation, seq_lengths_val = DataReader.read_from_file('MousePred/Data/validationData.txt')
    max_seq = np.max(seq_lengths_train + seq_lengths_val)
    min_seq = np.min(seq_lengths_train + seq_lengths_val)
    neural_network = NeuralNetwork(min_seq, max_seq, 'lol')
    for _ in range(2000):
        a = 0
        loss = 0
        for data_in in my_data:
            if len(data_in) <= 2:
                continue
            last_point = data_in[len(data_in) - 1]
            first_point = data_in[0]

            first_last = np.hstack((np.array(first_point)[0:2], np.array(last_point)[0:2]))
            distance = np.sqrt(np.power(first_last[0] - last_point[0], 2) + np.power(first_last[1] - last_point[1], 2))
            l, b = neural_network.train(np.array([distance]), len(data_in))
            a += 1
            loss += abs(l)

        print(loss/a)
    # Should be 0 , 1
       #print(neural_network.think([[0, 0, 1]]))