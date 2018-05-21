import csv
import numpy as np
import collections

class DataReader:
    def __init__(self, batch_size, filename, normalize=False):
        self.filename = filename
        self.batch_size = batch_size
        self.normalize = normalize
        self.iteration = 0

    def get_batch_bot_index(self):
        if self.iteration == 0:
            return self.batch_size * self.iteration + 1
        else:
            return self.batch_size * self.iteration + 1

    def get_batch_top_index(self):
        if self.iteration == 0:
            return self.batch_size + self.batch_size * self.iteration + 1
        else:
            return self.batch_size + self.batch_size * self.iteration + 1

    def increase_iteration_count(self):
        self.iteration += 1

    @staticmethod
    def get_reader_col_count(reader):
        return len(next(reader))

    def get_next_batch(self):
        n, row_count = 0, 0
        with open(self.filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            col_count = self.get_reader_col_count(spamreader)
            csvfile.seek(0)
            data_batch = np.zeros((self.batch_size, col_count))
            bot = self.get_batch_bot_index()
            top = self.get_batch_top_index()
            for row in spamreader:
                if bot <= n < top:
                    data_batch[n-bot] = row[0:col_count]
                    row_count += 1
                n += 1
                if n > top:
                    break

        if self.normalize:
            data_batch = self.scale_to_one(data_batch)

        self.increase_iteration_count()
        MetaData = collections.namedtuple('MetaData', ['col_count', 'row_count'])
        ReadData = collections.namedtuple('ReadData', ['data', 'meta_data'])
        result = ReadData(data_batch, MetaData(col_count, row_count))
        return result

    def scale_to_one(self, data):
        temp_min = data.min(axis=0)
        temp_max = data.max(axis=0)
        np.set_printoptions(threshold=5)
        temp = (data-temp_min)/(temp_max-temp_min)
        self.min = temp_min
        self.max = temp_max
        return temp

    def scale_from_one(self, data, is_dict=False, index=-1):
        if is_dict:
            data = self.dict_to_array(data)
        if index < 0:
            temp = (data*(self.max-self.min)) + self.min
        else:
            temp = (data*(self.max[index]-self.min[index])) + self.min[index]
        return temp


    def dict_to_array(self, dict):
        arr = np.zeros(len(dict))
        for i in range(len(dict)):
            arr[i] = dict[i][0]

        return arr

    def read_text_one_hot(self, filename):
        data = open('Eminem.txt', 'r').read()
        return data

    @staticmethod
    def read_from_file(filename):
        result = []
        seq_lengths = []
        seq_length = 0
        with open(filename) as f:
            lines = f.readlines()
        points = []
        for line in lines:
            parts = line.split(' ')
            if len(parts) <= 2:
                if len(points) != 0:
                    result.append(points)
                    seq_lengths.append(seq_length)
                points = []
                seq_length = 0
                continue
            x = float(parts[0]) / 1000
            y = float(parts[1]) / 1000
            if len(parts) <= 2:
                print('o')
            #time = float(parts[2]) / 1e8
            points.append([x, y])
            seq_length += 1

        '''
        data_min = 1e20
        data_max = -1e20
        for i in range(len(result)):
            for j in range(1, len(result[i])):
                if result[i][j][2] > 1:
                    result[i][j][2] /= 3
                if data_max < result[i][j][2]:
                    data_max = result[i][j][2]
                if data_min > result[i][j][2]:
                    data_min = result[i][j][2]

        for i in range(len(result)):
            for j in range(1, len(result[i])):
                result[i][j][2] = (result[i][j][2] - data_min) / (data_max - data_min)
        '''

        return result, seq_lengths