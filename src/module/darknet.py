import numpy as np
import os
import tensorflow as tf
import src.net_core.darknet as darknet

class Darknet19WeightReader(object):
    def __init__(self, weight_file_path):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file_path, dtype='float32')
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    def reset(self):
        self.offset = 4

class Darknet19WeightLoader(object):
    def __init__(self, weight_file_path, name='nolbo'):
        self._name = name
        self._buildModel()
        self._weightReader = Darknet19WeightReader(weight_file_path)
        self._ReadWeights()

    def _buildModel(self):
        print('build Models...')
        self.model = darknet.Darknet19(name=self._name+'_backbone', activation='lrelu')
        self.model.summary()

    def _ReadWeights(self):
        print("all_weights.shape={}".format(self._weightReader.all_weights.shape))
        nb_conv = 18
        for i in range(1, nb_conv+1):
            conv_layer = self.model.get_layer('conv_' + str(i))
            print('layer:',i)
            if i < nb_conv+1:
                norm_layer = self.model.get_layer('norm_' + str(i))
                size = np.prod(norm_layer.get_weights()[0].shape)
                beta = self._weightReader.read_bytes(size)
                gamma = self._weightReader.read_bytes(size)
                mean = self._weightReader.read_bytes(size)
                var = self._weightReader.read_bytes(size)
                print('norm:', beta, gamma, mean, var)

                weights = norm_layer.set_weights([gamma, beta, mean, var])

            if len(conv_layer.get_weights())>1:
                bias = self._weightReader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                kernel = self._weightReader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                conv_layer.set_weights([kernel, bias])
                print('cnn:', kernel, bias)
            else:
                kernel = self._weightReader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2,3,1,0])
                conv_layer.set_weights([kernel])
                print('cnn:', kernel)

    def saveModel(self, save_path):
        self.model.save_weights(os.path.join(save_path, self._name+'_backbone'))

# a = Darknet19WeightLoader(weight_file_path='../../weights/yolov2/yolov2.weights')
# a.saveModel('../../weights/yolov2/')