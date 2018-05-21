import numpy as np
import csv
from random import shuffle
from Reader import DataReader
from NNMath import NNMath
from NeuralNetwork import NeuralNetwork
from os import getcwd, path, makedirs
import sys
import matplotlib.pyplot as plt
import pathlib


class LSTM:
    def __init__(self, input_size, num_hidden_units, output_size, folder, import_data = False):
        self.num_hidden_units = num_hidden_units
        self.output_size = output_size
        self.input_size = input_size
        self.combined = num_hidden_units + input_size
        self.learning_rate = 0.1
        self.iteration_count = 11
        self.loss_history = []
        self.valid_loss_history = []
        self.importWeights = import_data
        self.exportWeights = True
        self.import_weights_dir = ""
        self.export_weights_dir = ""
        self.export_every = 5
        self.main_dir = getcwd() + '/MousePred/Models/'
        if not path.exists(self.main_dir + folder +'/'):
            makedirs(self.main_dir + folder +'/')

        self.main_dir = getcwd() + '/MousePred/Models/' + folder + '/'
        self.loss_history = []
        self.valid_loss_history = []
        self.wa, self.wf, self.wo, self.wi, self.wy, self.bf, self.bo, self.ba, self.bi, self.by = self.init_weights()
        self.mdUf, self.mdUo, self.mdUi, self.mdUs, self.mdUy = np.zeros_like(self.wf), np.zeros_like(self.wo), np.zeros_like(
            self.wi), np.zeros_like(
            self.wa), np.zeros_like(self.wy)
        self.mdbo, self.mdbi, self.mdbs, self.mdbf, self.mdby = np.zeros_like(self.bo), np.zeros_like(self.bi), np.zeros_like(
            self.ba), np.zeros_like(
            self.bf), np.zeros_like(self.by)
        self.learning_rate_div = 1


    def init_weights(self):
        num_hidden_units = self.num_hidden_units
        combined = self.combined
        output_size = self.output_size
        wa = np.random.rand(num_hidden_units, combined) / np.sqrt(combined / 2.) * 0.01
        wf = np.random.rand(num_hidden_units, combined) / np.sqrt(combined / 2.) * 0.01
        wo = np.random.rand(num_hidden_units, combined) / np.sqrt(combined / 2.) * 0.01
        wi = np.random.rand(num_hidden_units, combined) / np.sqrt(combined / 2.) * 0.01
        wy = np.random.rand(output_size, num_hidden_units) / np.sqrt(num_hidden_units / 2.) * 0.01

        bf = np.zeros(num_hidden_units)
        bo = np.zeros(num_hidden_units)
        ba = np.zeros(num_hidden_units)
        bi = np.zeros(num_hidden_units)
        by = np.zeros(output_size)

        return wa, wf, wo, wi, wy, bf, bo, ba, bi, by

    def export_weights(self, dir=""):
        np.savetxt(dir + "wa.csv", self.wa, delimiter=",")
        np.savetxt(dir + "wf.csv", self.wf, delimiter=",")
        np.savetxt(dir + "wo.csv", self.wo, delimiter=",")
        np.savetxt(dir + "wi.csv", self.wi, delimiter=",")
        np.savetxt(dir + "wy.csv", self.wy, delimiter=",")

        np.savetxt(dir + "ba.csv", self.ba, delimiter=",")
        np.savetxt(dir + "bf.csv", self.bf, delimiter=",")
        np.savetxt(dir + "bo.csv", self.bo, delimiter=",")
        np.savetxt(dir + "bi.csv", self.bi, delimiter=",")
        np.savetxt(dir + "by.csv", self.by, delimiter=",")

    def import_weights(self, dir=""):
        self.wa = np.array(list(csv.reader(open(dir + "wa.csv", "r"), delimiter=","))).astype(float)
        self.wf = np.array(list(csv.reader(open(dir + "wf.csv", "r"), delimiter=","))).astype(float)
        self.wo = np.array(list(csv.reader(open(dir + "wo.csv", "r"), delimiter=","))).astype(float)
        self.wi = np.array(list(csv.reader(open(dir + "wi.csv", "r"), delimiter=","))).astype(float)
        self.wy = np.array(list(csv.reader(open(dir + "wy.csv", "r"), delimiter=","))).astype(float)

        self.ba = np.array(list(csv.reader(open(dir + "ba.csv", "r"), delimiter=","))).astype(float).T[0]
        self.bf = np.array(list(csv.reader(open(dir + "bf.csv", "r"), delimiter=","))).astype(float).T[0]
        self.bo = np.array(list(csv.reader(open(dir + "bo.csv", "r"), delimiter=","))).astype(float).T[0]
        self.bi = np.array(list(csv.reader(open(dir + "bi.csv", "r"), delimiter=","))).astype(float).T[0]
        self.by = np.array(list(csv.reader(open(dir + "by.csv", "r"), delimiter=","))).astype(float).T[0]

    def loss_function(self, inputs, hp, sp, y, last, use_softmax=False):
        g_s, g_f, g_o, g_i, g_a, g_y, x, h = {}, {}, {}, {}, {}, {}, {}, {}

        h[-1] = np.atleast_2d(hp)
        g_s[-1] = np.atleast_2d(sp)
        loss = 0
        y = np.asarray(y)
        for t in range(len(y)):
            x[t] = np.atleast_2d(np.zeros((self.input_size)))
            temp_in = np.hstack((inputs[t], np.array([last[0], last[1]]), NNMath.scale_to_one(np.array([len(y)-t]), 0, self.max_seq)))
            x[t] = temp_in
            x[t] = np.hstack((np.atleast_2d(x[t]), h[t - 1]))[0]

            g_f[t] = NNMath.sigmoid(self.bf + np.dot(self.wf, x[t]))
            g_i[t] = NNMath.sigmoid(self.bi + np.dot(self.wi, x[t]))
            g_a[t] = NNMath.tanh(self.ba + np.dot(self.wa, x[t]))
            g_o[t] = NNMath.sigmoid(self.bo + np.dot(self.wo, x[t]))
            g_s[t] = g_f[t] * g_s[t - 1] + g_i[t] * g_a[t]

            h[t] = np.tanh(g_s[t]) * g_o[t]
            if use_softmax:
                g_y[t] = NNMath.soft_max(np.dot(self.wy, h[t][0]) + self.by)
            else:
                g_y[t] = NNMath.sigmoid(np.dot(self.wy, h[t][0]) + self.by)
            loss += abs(NNMath.loss_multi(g_y[t], y[t]))

        duo, dbo = np.zeros_like(self.wo), np.zeros_like(self.bo)
        dua, dba = np.zeros_like(self.wa), np.zeros_like(self.ba)
        duf, dbf = np.zeros_like(self.wf), np.zeros_like(self.bf)
        dui, dbi = np.zeros_like(self.wi), np.zeros_like(self.bi)
        duy, dby = np.zeros_like(self.wy), np.zeros_like(self.by)
        delta_out = np.zeros(self.num_hidden_units)
        delta_s_future = np.zeros(self.num_hidden_units)

        for t in reversed(range(len(y))):
            pred = g_y[t]
            if use_softmax:
                label = np.zeros_like(pred)
                label[y[t]] = 1
                diff = (pred - label)
            else:
                diff = (pred - y[t])

            L = diff
            L = np.atleast_2d(L)
            duy += np.dot(L.T, h[t])
            dby += L[0]
            dout = np.dot(L, self.wy) + delta_out
            dstate = dout * g_o[t] * NNMath.tanh_derivative2(g_s[t]) + delta_s_future
            df = dstate * g_s[t - 1] * NNMath.sigmoid_derivative(g_f[t])
            di = dstate * g_a[t] * NNMath.sigmoid_derivative(g_i[t])
            da = dstate * g_i[t] * NNMath.tanh_derivative(g_a[t])
            do = dout * np.tanh(g_s[t]) * NNMath.sigmoid_derivative(g_o[t])

            delta_out = np.dot(self.wi.T, di[0]) \
                        + np.dot(self.wo.T, do[0]) \
                        + np.dot(self.wf.T, df[0]) \
                        + np.dot(self.wa.T, da[0])

            delta_out = delta_out[self.input_size:]
            delta_s_future = dstate * g_f[t]

            duo += np.outer(do, x[t])
            dui += np.outer(di, x[t])
            duf += np.outer(df, x[t])
            dua += np.outer(da, x[t])

            dbo += do[0]
            dba += da[0]
            dbf += df[0]
            dbi += di[0]

        # Gradient clipping
        for dparam in [duf, duo, dui, dua, duy, dbo, dbi, dba, dbf, dby]:
            np.clip(dparam, -1, 1, out=dparam)

        return loss, duf, duo, dui, dua, duy, dbo, dbi, dba, dbf, dby, h[len(y) - 1], g_s[len(y) - 1], g_y

    def feed_forward(self, hp, sprev, x, targets, remaining_len, sampling = False):
        x = np.hstack((x, np.array(NNMath.scale_to_one(remaining_len, 0, self.max_seq)), hp))

        g_f = NNMath.sigmoid(self.bf + np.dot(self.wf, x))
        g_i = NNMath.sigmoid(self.bi + np.dot(self.wi, x))
        g_a = NNMath.tanh(self.ba + np.dot(self.wa, x))
        g_o = NNMath.sigmoid(self.bo + np.dot(self.wo, x))

        g_s = g_f * sprev + g_i * g_a
        sprev = g_s
        hp = np.tanh(g_s) * g_o
        if sampling:
            adjustx = 0
            adjusty = 0
            if np.random.randint(0, 10) <= 5 and remaining_len >= 4:
                adjustx = np.random.normal(0, 0.001)
                adjusty = np.random.normal(0, 0.001)
            g_y = NNMath.sigmoid(np.dot(self.wy, hp) + self.by)
            g_y[0] += adjustx
            g_y[1] += adjusty
        else:
            g_y = NNMath.sigmoid(np.dot(self.wy, hp) + self.by)

        loss = NNMath.loss_multi(g_y, targets)

        return loss, sprev, hp, g_y

    def validation_loss(self, inputs, targets, last):
        hp = np.zeros(self.num_hidden_units)
        sprev = np.zeros(self.num_hidden_units)
        loss = 0
        for t in range(len(targets)):
            temp_in = (np.hstack((inputs[t], np.array([last[0], last[1]]))))
            received_loss, sprev, hp, g_y = self.feed_forward(hp, sprev, temp_in, targets[t], NNMath.scale_to_one(len(targets)-t, 0, self.max_seq))
            loss += abs(received_loss)

        return loss

    def get_validation_loss(self, my_data_validation):
        val_loss = 0
        a = 0
        for validataion_in in my_data_validation:
            if len(validataion_in) <= 2:
                continue

            last_point = validataion_in[len(validataion_in) - 1]

            npdata = np.array(validataion_in)
            #npdata[0][2] = 0

            inputs = npdata[0:len(npdata) - 1]
            targets = npdata[1:len(npdata)]

            val_loss += self.validation_loss(inputs, targets, last_point)
            a += 1
        val_loss = abs(val_loss / a)
        return val_loss

    def sample(self, inputs, last, sequence_len):
        hp = np.zeros(self.num_hidden_units)
        sprev = np.zeros(self.num_hidden_units)
        g_y = [0, 0]
        eps = 0.01
        x = inputs
        t = 0
        rez = []
        a = 0
        while True:
            if sequence_len - a == 0:
                break
            if abs(g_y[0] - last[0]) < eps and abs(g_y[1] - last[1]) < eps:
                break
            if t >= 100:
                break
            temp_in = np.hstack((x, np.array([last[0], last[1]])))
            received_loss, sprev, hp, g_y = self.feed_forward(hp, sprev, temp_in, [0, 0], sequence_len-a, True)

            x = g_y
            t += 1
            a += 1
            rez.append(g_y)

        return rez

    def calculate_sequence(self, distance):
        if distance <= 0.015:
            return 3
        return np.ceil(distance * 2 / 0.01)

    def write_sample(self):
        file = open("testfile.txt", "w")
        first_last = np.array([0.12, 0.15, 0.90, 0.75])
        distance = np.sqrt(np.power(first_last[0] - first_last[2], 2) + np.power(first_last[1] - first_last[3], 2))
        seqLen = int(round(self.calculate_sequence(distance)))#int(round(self.Length_pred.think(np.array([distance]))[0]))
        print(distance, seqLen)
        samp = self.sample(first_last[0:2], first_last[2:4], seqLen)
        for s in samp:
            file.write("%f %f \n" % (s[0], s[1]))

        file.write("Break \n")
        first_last = np.array([0.90, 0.15, 0.18, 0.32])
        distance = np.sqrt(np.power(first_last[0] - first_last[2], 2) + np.power(first_last[1] - first_last[3], 2))
        seqLen = int(round(self.calculate_sequence(distance)))  # int(round(self.Length_pred.think(np.array([distance]))[0]))
        print(distance, seqLen)
        samp = self.sample(first_last[0:2], first_last[2:4], seqLen)
        for s in samp:
            file.write("%f %f \n" % (s[0], s[1]))

        file.write("Break \n")
        first_last = np.array([0.22, 0.263, 0.774, 0.863])
        distance = np.sqrt(np.power(first_last[0] - first_last[2], 2) + np.power(first_last[1] - first_last[3], 2))
        seqLen = int(round(self.calculate_sequence(distance)))  # int(round(self.Length_pred.think(np.array([distance]))[0]))
        print(distance, seqLen)
        samp = self.sample(first_last[0:2], first_last[2:4], seqLen)
        for s in samp:
            file.write("%f %f \n" % (s[0], s[1]))

        file.write("Break \n")
        first_last = np.array([0.53, 0.263, 0.532, 0.230])
        distance = np.sqrt(np.power(first_last[0] - first_last[2], 2) + np.power(first_last[1] - first_last[3], 2))
        seqLen = int(round(self.calculate_sequence(distance)))  # int(round(self.Length_pred.think(np.array([distance]))[0]))
        print(distance, seqLen)
        samp = self.sample(first_last[0:2], first_last[2:4], seqLen)
        for s in samp:
            file.write("%f %f \n" % (s[0], s[1]))

        file.close()

    def weight_update(self, dUf, dUo, dUi, dUs, dUy, dbo, dbi, dbs, dbf, dby):
        # adagrad update
        for param, dparam, mem in zip(
                [self.wf, self.wo, self.wi, self.wa, self.wy, self.bo, self.bi, self.ba, self.bf, self.by],
                [dUf, dUo, dUi, dUs, dUy, dbo, dbi, dbs, dbf, dby],
                [self.mdUf, self.mdUo, self.mdUi, self.mdUs, self.mdUy, self.mdbo, self.mdbi, self.mdbs, self.mdbf, self.mdby]):
            mem += dparam * dparam
            param += -self.learning_rate / self.learning_rate_div * dparam / np.sqrt(
                mem + 1e-8)



    def train(self):
        n = 0
        my_data, seq_lengths_train = DataReader.read_from_file('MousePred/Data/MouseData.txt')
        my_data_validation, seq_lengths_val = DataReader.read_from_file('MousePred/Data/validationData.txt')
        self.max_seq = 290
        self.min_seq = 3
        '''
        self.max_seq = np.max(seq_lengths_train + seq_lengths_val)
        self.min_seq = np.min(seq_lengths_train + seq_lengths_val)
        self.Length_pred = NeuralNetwork(self.min_seq, self.max_seq, 'lol')
        '''
        c = 0
        eps = 0.001
        LossHist = {}

        if self.importWeights:
            self.import_weights(self.main_dir)
            #self.Length_pred.export_weights()

        print('iter %d, validation_loss: %f' % (n, self.get_validation_loss(my_data_validation)))
        while n <= self.iteration_count:
            avgLoss = 0
            shuffle(my_data)
            ac = 0
            loss_nn = 0
            if c == 3:
                c = 0
            for data_in in my_data:
                if len(data_in) <= 2:
                    continue
                last_point = data_in[len(data_in) - 1]
                first_point = data_in[0]

                first_last = np.hstack((np.array(first_point)[0:2], np.array(last_point)[0:2]))
                #nnloss, seq = self.Length_pred.train(np.array([distance]), len(data_in))
                loss_nn += 0
                hprev = np.zeros(self.num_hidden_units)  # reset LSTM memory
                sprev = np.zeros(self.num_hidden_units)

                npdata = np.array(data_in)
                #npdata[0][2] = 0

                inputs = npdata[0:len(npdata) - 1]
                targets = npdata[1:len(npdata)]
                loss, dUf, dUo, dUi, dUs, dUy, dbo, dbi, dbs, dbf, dby, hprev, sprev, rez = self.loss_function(inputs, hprev,
                                                                                                          sprev,
                                                                                                          targets,
                                                                                                          last_point,
                                                                                                          )

                self.weight_update(dUf, dUo, dUi, dUs, dUy, dbo, dbi, dbs, dbf, dby)
                ac += 1
                avgLoss += loss

            LossHist[c] = avgLoss / ac
            c += 1
            print(n)
            if n % self.export_every == 0 and n > 0 and self.exportWeights:
                print(self.main_dir)
                self.export_weights(self.main_dir)
                #self.Length_pred.export_weights()
            if n % 5 == 0 and n != 0:
                for i in range(len(data_in) - 1):
                    print(targets[i])
                    print(rez[i])
                    print('-------------------')
                self.write_sample()

            valid_loss = self.get_validation_loss(my_data_validation)
            self.loss_history.append(avgLoss / (len(my_data) - 1))
            self.valid_loss_history.append(valid_loss)

            print(self.loss_history)
            print(self.valid_loss_history)
            print('iter %d, loss: %f' % (n, avgLoss / (len(my_data) - 1)))
            #print('iter %d, NNloss: %f, Sclaed: %f' % (n, loss_nn / (len(my_data) - 1), NNMath.scale_from_one(loss_nn/ (len(my_data) - 1), self.min_seq, self.max_seq)))
            print('iter %d, validation_loss: %f' % (n, valid_loss))
            if n != 0 and n > 4:
                print(abs(LossHist[0] - LossHist[2]))
                if abs(LossHist[0] - LossHist[2]) < eps:
                    print('change learning rate')
                    print(LossHist[0] - LossHist[2])
                    self.learning_rate_div *= 2
                    eps /= 5

            n += 1  # iteration counter

    def print_to_file(self, input, output):
        file = open("testfilecmd.txt", "w")
        distance = self.calculate_distance(input, output)
        seqLen = int(round(self.calculate_sequence(distance)))  # int(round(self.Length_pred.think(np.array([distance]))[0]))
        print('distance %f, seq_length: %d' % (distance,  seqLen))
        samp = self.sample(input, output, seqLen)
        plt.axis([0, 1, 0, 1])

        plt.plot(np.array(samp)[:, 0], np.array(samp)[:, 1])
        plt.scatter(output[0], output[1])
        for s in samp:
            file.write("%f %f \n" % (s[0], s[1]))
        plt.show()
        file.close()


    @staticmethod
    def calculate_distance(starting_point, end_point):
        return np.sqrt(np.power(starting_point[0]-end_point[0], 2) + np.power(starting_point[1]-end_point[1], 2))

a = LSTM(5, 100, 2, 'lol2')

if __name__ == '__main__':

    if len(sys.argv) <= 3:
        a.train()
        exit(0)

    sys.stderr.write(str(sys.argv))
    print('hi')
    a.max_seq = 290
    a.min_seq = 3
    a.import_weights(a.main_dir)
    starting_point = np.array([float(sys.argv[1]), float(sys.argv[2])])#    starting_point = np.array([float(sys.argv[1]), float(sys.argv[2])])#

    end_point = np.array([float(sys.argv[3]), float(sys.argv[4])]) #np.array([0.131, 0.45]) #
    a.print_to_file(starting_point, end_point)
    #print(a.sample(starting_point, end_point, a.calculate_distance(starting_point, end_point)))


