import numpy as np
import os, re, sys
from scipy.ndimage import zoom

class dataLoader(object):
    def __init__(self, data_path, partial_num=30, objectNet=False):
        self.epoch = 0
        self._data_path = data_path
        self._partial_num = partial_num
        self.batchStart = 0

        self._vox3DData = []
        self._classList = []
        self._instList = []
        self.dataLength = 0
        self._dataIdx = None
        self._objectNet = objectNet

        self._loadData()
        self._dataIdxShuffle()

    def _voxBatch_covert_axis_32to64(self, vox32_batch):
        vox32_batch_convert_axis = np.transpose(vox32_batch, [0, 3, 1, 2, 4])  # original modelnet axis -> pascal3D axis
        vox64_batch = zoom(vox32_batch_convert_axis, (1, 2, 2, 2, 1))
        vox64_batch = np.where(vox64_batch > 0.5, np.ones_like(vox64_batch), np.zeros(vox64_batch))
        return vox64_batch

    def _loadData(self):
        print('load data...')
        self._vox3DData = []
        self._classList = []
        self._instList = []
        for i in range(self._partial_num):  # train set
            voxTemp = np.load(os.path.join(self._data_path, '32to64_4rot_64sqr', 'train', str(i) + 'Full.npy'))
            classTemp = np.load(os.path.join(self._data_path, '32to64_4rot_64sqr', 'train', str(i) + 'Class.npy'))
            instTemp = np.load(os.path.join(self._data_path, '32to64_4rot_64sqr', 'train', str(i) + 'Inst.npy'))
            self._vox3DData.append(voxTemp)
            self._classList.append(classTemp)
            self._instList.append(instTemp)
            sys.stdout.write("train data:{:02d}/{:02d}   \r".format(i+1, self._partial_num))
        print('')

        for i in range(5):  # test set
            voxTemp = np.load(os.path.join(self._data_path, '32to64_4rot_64sqr', 'test', str(i) + 'Full.npy'))
            classTemp = np.load(os.path.join(self._data_path, '32to64_4rot_64sqr', 'test', str(i) + 'Class.npy'))
            instTemp = np.load(os.path.join(self._data_path, '32to64_4rot_64sqr', 'test', str(i) + 'Inst.npy'))
            self._vox3DData.append(voxTemp)
            self._classList.append(classTemp)
            self._instList.append(instTemp)
            sys.stdout.write("test data:{:02d}/{:02d}   \r".format(i+1, 5))
        print('')

        if self._objectNet:
            data_path = '/media/yonsei/4TB_HDD/dataset/ObjectNet3D/CAD_training_data_numpy/'
            objFolderList = os.listdir(data_path)
            for i, objFolder in enumerate(objFolderList):
                if os.path.isdir(os.path.join(data_path, objFolder)):
                    npyPath = os.path.join(data_path, objFolder, 'vox3DOrg.npy')
                    shape3D = np.load(npyPath)
                    if len(shape3D)>0:
                        self._vox3DData.append(shape3D)
                        classTemp = np.zeros([len(shape3D), self._classList[0].shape[-1]])
                        instTemp = np.zeros([len(shape3D), self._instList[0].shape[-1]])
                        self._classList.append(classTemp)
                        self._instList.append(instTemp)
                        sys.stdout.write("objectNet data:{:02d}/{:02d}   \r".format(i + 1, len(objFolderList)))
            print('')

        self._vox3DData = np.concatenate(self._vox3DData, axis=0)
        self._classList = np.concatenate(self._classList, axis=0)
        self._instList = np.concatenate(self._instList, axis=0)


        # # take only forward directional objects
        # data_temp_length = len(self._vox3DData)
        # self._vox3DData = self._vox3DData[[4*i for i in range(int(data_temp_length/4))]]
        # self._classList = self._classList[[4 * i for i in range(int(data_temp_length / 4))]]
        # self._instList = self._instList[[4 * i for i in range(int(data_temp_length / 4))]]

        # print(self._vox3DData.shape, self._classList.shape, self._instList.shape)
        self.dataLength = len(self._vox3DData)
        self._dataIdx = [i for i in range(self.dataLength)]
        print('done!')

    def _dataIdxShuffle(self):
        np.random.shuffle(self._dataIdx)
        self.batchStart = 0

    def getNextBatch(self, batchSize=32):
        if self.batchStart + batchSize > self.dataLength:
            self.epoch += 1
            self._dataIdxShuffle()
        dataStart = self.batchStart
        dataEnd = self.batchStart + batchSize
        self.batchStart += batchSize
        dataList = self._dataIdx[dataStart:dataEnd]
        # input_images = self._voxBatch_covert_axis_32to64(vox32_batch=(self._vox3DData[dataList]).astype('float'))
        input_images = (self._vox3DData[dataList]).astype('float32')
        class_list = (self._classList[dataList]).astype('float32')
        inst_list = (self._instList[dataList]).astype('float32')
        batch_dict = {
            'input_images': input_images,
            'class_list': class_list,
            'inst_list': inst_list,
        }
        return batch_dict
