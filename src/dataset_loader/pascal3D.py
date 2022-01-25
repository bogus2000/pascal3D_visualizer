import scipy.io
import os
import numpy as np
import cv2
from src.dataset_loader.datasetUtils import *
from src.module.function import kmeans_IoU
import src.dataset_loader.datasetUtils as datasetUtils
import ast

category_to_index = {
    'aeroplane' : 1,
    'bed' : 2,
    'bench' : 3,
    'bicycle' : 4,
    'boat' : 5,
    'bookshelf' : 6,
    'bottle' : 7,
    'bus' : 8,
    'cabinet' : 9,
    'can' : 10,
    'cap' : 11,
    'car' : 12,
    'chair' : 13,
    'computer' : 14,
    'cup' : 15,
    'desk' : 16,
    'table' : 17,
    'door' : 18,
    'fire_extinguisher' : 19,
    'jar' : 20,
    'keyboard' : 21,
    'laptop' : 22,
    'microwave' : 23,
    'motorbike' : 24,
    'mouse' : 25,
    'piano' : 26,
    'pillow' : 27,
    'printer' : 28,
    'refrigerator' : 29,
    'rode_pole' : 30,
    'sofa' : 31,
    'speaker' : 32,
    'suitcase' : 33,
    'teapot' : 34,
    'toilet' : 35,
    'train' : 36,
    'trash_bin' : 37,
    'bathtub' : 38,
    'tvmonitor' : 39,
    'wardrobe' : 40,
}

# average image size of pascal3D : (508.54, 404.47)
class Pascal3DMultiObject(object):
    def __init__(self,
                 imageSize=(640,480),
                 gridSize=(20,15),
                 predNumPerGrid=5,
                 Pascal3DDataPath=None,
                 trainOrVal='train',
                 isTrain=True,
                 ):
        self.dataStart = 0
        self.dataLength = 0
        self.epoch = 0

        self._imageSize = imageSize
        self._gridSize = gridSize
        self._predNumPerGrid = predNumPerGrid
        self._Pascal3DDataPath = Pascal3DDataPath
        self._trainOrVal = trainOrVal
        self._isTrain = isTrain
        self._dataPathList = []
        self._CAD3DShapes = None

        print('set pascal3d dataset...')
        self._getTrainList()
        self._loadDataPath()
        self._createDict()
        # self._load3DShapes()
        try:
            self._loadPrior()
        except:
            self.getKMeansBBoxIoU()
            self._loadPrior()
        self._dataPathShuffle()

    def _getTrainList(self):
        print('set train or val list...')
        self._Pascal3DTrainList = []
        datasetList = os.listdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/'))
        for datasetName in datasetList:
            if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/', datasetName)):
                txtFileList = os.listdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/',datasetName))
                for txtFileName in txtFileList:
                    className = txtFileName.split('.')[0].split('_')[0]
                    trainval = txtFileName.split('.')[0].split('_')[-1]
                    if trainval==self._trainOrVal:
                        with open(os.path.join(self._Pascal3DDataPath, 'Image_sets/',datasetName,txtFileName)) as txtFilePointer:
                            dataPointList = txtFilePointer.readlines()
                            for i, dataPoint in enumerate(dataPointList):
                                if datasetName == 'pascal':
                                    dp = dataPoint.split('\n')[0].split(' ')[0]
                                    isTrue = int(dataPoint.split('\n')[0].split(' ')[-1])
                                    if int(isTrue)==1:
                                        self._Pascal3DTrainList.append(dp)
                                else:
                                    dp = dataPoint.split('\n')[0].split(' ')[0]
                                    self._Pascal3DTrainList.append(dp)
        print('done!')

    def _createDict(self):
        print( 'create dict...')
        self._classDict = dict()
        self._instDict = dict()
        self._instNumMax = 0
        if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'CAD')):
            classList = os.listdir(os.path.join(self._Pascal3DDataPath, 'CAD'))
            classList.sort(key=datasetUtils.natural_keys)
            categoryIdx = 0
            for className in classList:
                if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'CAD', className)):
                    self._classDict[className] = categoryIdx
                    categoryIdx += 1
                    if className not in self._instDict:
                        self._instDict[className] = dict()
                    CADModelList = os.listdir(os.path.join(self._Pascal3DDataPath, 'CAD', className))
                    CADModelList.sort(key=datasetUtils.natural_keys)
                    instIdx = 0
                    for CADModel in CADModelList:
                        if CADModel.endswith('.pcd'):
                            CADModelPath = os.path.join('CAD', className, os.path.splitext(CADModel)[0])
                            if CADModelPath not in self._instDict[className]:
                                self._instDict[className][CADModelPath] = instIdx
                                instIdx += 1
                    self._instNumMax = np.max((self._instNumMax, instIdx))
        print('category_num:', len(self._classDict))
        print('inst_num_max:', self._instNumMax)
        print('dict ready!')

    def _loadDataPath(self):
        print('load datapoint path...')
        datasetList = os.listdir(os.path.join(self._Pascal3DDataPath, 'training_data'))
        for datasetName in datasetList:
            if datasetName == 'imagenet' or datasetName == 'pascal':
                dataPointList = os.listdir(os.path.join(self._Pascal3DDataPath, 'training_data', datasetName))
                for dataPointName in dataPointList:
                    if dataPointName in self._Pascal3DTrainList:
                        if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'training_data', datasetName, dataPointName)):
                            dataPointPath = os.path.join(self._Pascal3DDataPath, 'training_data', datasetName, dataPointName)
                            self._dataPathList.append(dataPointPath)
        self._dataPathList = np.array(self._dataPathList)
        self.dataLength = len(self._dataPathList)
        print('done!')

    def _dataPathShuffle(self):
        print('')
        print('data path shuffle...')
        self.dataStart = 0
        np.random.shuffle(self._dataPathList)
        self.dataLength = len(self._dataPathList)
        print('done! : ' + str(self.dataLength))

    # def _load3DShapes(self):
    #     print('load 3d shapes for pascal3d...')
    #     self._CAD3DShapes = []
    #     CADModelList = os.listdir(os.path.join(self._Pascal3DDataPath, 'CAD', 'car'))
    #     CADModelList.sort()
    #     for CADModel in CADModelList:
    #         if CADModel.split(".")[-1] == 'npy':
    #             shape = np.load(os.path.join(self._Pascal3DDataPath, 'CAD', 'car', CADModel)).reshape(64, 64, 64, 1)
    #             self._CAD3DShapes.append(shape)
    #     self._CAD3DShapes = np.array(self._CAD3DShapes)
    #     self._CAD3DShapes = np.where(self._CAD3DShapes>0, 1.0, 0.0)
    #     print('done!')

    def getKMeansBBoxIoU(self, k=None, max_iter=1000):
        if k == None:
            k = self._predNumPerGrid
        bboxHW_normalized = []
        for i, dataPath in enumerate(self._dataPathList):
            print(i, len(self._dataPathList))
            objFolderList = os.listdir(dataPath)
            objList = []
            for objFolder in objFolderList:
                objFolder = str(objFolder.decode('utf-8'))
                if os.path.isdir(os.path.join(dataPath, objFolder)):
                    objInfoTXT = os.path.join(dataPath, objFolder, 'objInfo.txt')
                    with open(objInfoTXT) as objInfoPointer:
                        objInfo = objInfoPointer.readline()
                        objList.append(objInfo)
            if len(objList)>0:
                imagePath = os.path.join(self._Pascal3DDataPath, objList[0].split(' ')[1])
                image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                imageRow, imageCol, _ = image.shape
                imageRow, imageCol = float(imageRow), float(imageCol)
                for objIndex, obj in enumerate(objList):
                    className, imagePath, CADModelPath, colMin, rowMin, colMax, rowMax, azimuth, elevation, inPlaneRot = obj.split(' ')
                    h = (float(rowMax) - float(rowMin)) / imageRow
                    w = (float(colMax) - float(colMin)) / imageCol
                    bboxHW_normalized.append([h,w])
        bboxHW_normalized = np.array(bboxHW_normalized)
        kmeans = kmeans_IoU(X=bboxHW_normalized, k=k, max_iter=max_iter)
        bboxMean, ktoc, dist = kmeans.kmeansSample()
        bboxMean = bboxMean[np.argsort(bboxMean[..., 0]*bboxMean[..., 1])]
        print(bboxMean)
        np.save('bboxMean.npy', bboxMean)

    def getImagePrior(self):
        imageSize = []
        imagePixelMean = np.zeros((3,))
        imagePixelStd = np.zeros((3,))
        for i, dataPath in enumerate(self._dataPathList):
            print(i, len(self._dataPathList))
            objFolderList = os.listdir(dataPath)
            objList = []
            for objFolder in objFolderList:
                objFolder = str(objFolder)
                print(dataPath, objFolder)
                if os.path.isdir(os.path.join(dataPath, objFolder)):
                    objInfoTXT = os.path.join(dataPath, objFolder, 'objInfo.txt')
                    with open(objInfoTXT) as objInfoPointer:
                        objInfo = objInfoPointer.readline()
                        objList.append(objInfo)
            if len(objList) > 0:
                imagePath = os.path.join(self._Pascal3DDataPath, objList[0].split(' ')[1])
                image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                imageRow, imageCol, _ = image.shape
                imageRow, imageCol = float(imageRow), float(imageCol)
                imagePixelSum_curr = np.sum(np.sum(image, axis=0), axis=0)
                imagePixelMean += imagePixelSum_curr
                imageSize.append([imageRow, imageCol])
        imageSize = np.array(imageSize)
        pixel_num = np.sum(imageSize[..., 0] * imageSize[..., 1])
        imagePixelMean = imagePixelMean / pixel_num
        for i, dataPath in enumerate(self._dataPathList):
            print(i, len(self._dataPathList))
            objFolderList = os.listdir(dataPath)
            objList = []
            for objFolder in objFolderList:
                objFolder = str(objFolder)
                if os.path.isdir(os.path.join(dataPath, objFolder)):
                    objInfoTXT = os.path.join(dataPath, objFolder, 'objInfo.txt')
                    with open(objInfoTXT) as objInfoPointer:
                        objInfo = objInfoPointer.readline()
                        objList.append(objInfo)
            if len(objList) > 0:
                imagePath = os.path.join(self._Pascal3DDataPath, objList[0].split(' ')[1])
                image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
                imagePixelStd += np.sum(np.sum(np.square(image - imagePixelMean), axis=0), axis=0)
        imagePixelStd = np.sqrt(imagePixelStd/pixel_num)
        imageSizeMean = np.mean(imageSize, axis=0)
        print(imagePixelMean)
        print(imagePixelStd)
        print(imageSizeMean)
        np.save('imagePixelMean.npy', imagePixelMean)
        np.save('imagePixelStd.npy', imagePixelStd)
        np.save('imageSizeMean.npy', imageSizeMean)

    def _loadPrior(self):
        self._imagePixelMean = np.load('./src/dataset_loader/imagePixelMean.npy')
        self._imagePixelStd = np.load('./src/dataset_loader/imagePixelStd.npy')
        self._imageSizeMean = np.load('./src/dataset_loader/imageSizeMean.npy')
        self._bboxHWMean = np.load('./src/dataset_loader/bboxMean.npy')

    def _dist_IoU(self, X, centres):
        k, cdim = centres.shape
        h_X, w_X = X
        X_area = h_X * w_X
        x_min_X, x_max_X, y_min_X, y_max_X = -w_X/2., w_X/2., -h_X/2., h_X/2.
        h_c, w_c = centres[:,0], centres[:,1]
        c_area = h_c * w_c
        x_min_c, x_max_c, y_min_c, y_max_c = -w_c/2., w_c/2., -h_c/2., h_c/2.
        dist = []
        for ik in range(k):
            intersection_x_min = np.max((x_min_X, x_min_c[ik]))
            intersection_y_min = np.max((y_min_X, y_min_c[ik]))
            intersection_x_max = np.min((x_max_X, x_max_c[ik]))
            intersection_y_max = np.min((y_max_X, y_max_c[ik]))
            intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
            IoU = intersection_area / (X_area + c_area[ik] - intersection_area + 1e-9)
            dist.append(1. - IoU)
        return np.array(dist)

    def _getOffset(self, batchSize):
        offsetX = np.transpose(np.reshape(
            np.array([np.arange(self._gridSize[0])]*self._gridSize[1]*self._predNumPerGrid),
            (self._predNumPerGrid, self._gridSize[1], self._gridSize[0])), (1,2,0))
        offsetX = np.tile(np.reshape(offsetX, (1,self._gridSize[1],self._gridSize[0],self._predNumPerGrid)),[batchSize,1,1,1])
        offsetY = np.transpose(np.reshape(
            np.array([np.arange(self._gridSize[1])]*self._gridSize[0]*self._predNumPerGrid),
            (self._predNumPerGrid, self._gridSize[0], self._gridSize[1])), (2,1,0))
        offsetY = np.tile(np.reshape(offsetY, (1,self._gridSize[1],self._gridSize[0],self._predNumPerGrid)),[batchSize,1,1,1])
        return offsetX.astype('float'), offsetY.astype('float')

    def getNextBatch(self, batchSizeof3DShape=32, imageSize=None, gridSize=None):
        if imageSize!=None:
            self._imageSize = imageSize
        if gridSize!=None:
            self._gridSize = gridSize
        inputImages, bboxHWImages, bboxXYImages, objnessImages, EulerRadImages = [], [], [], [], []
        outputImages, categoryList, instList, EulerRadList = [], [], [], []
        while len(outputImages)==0:
            for dataPath in self._dataPathList[self.dataStart:]:
                objFolderList = os.listdir(dataPath)
                np.random.shuffle(objFolderList)
                # objFolderList.sort()
                objSelectedList = []
                for objFolder in objFolderList:
                    objFolder = str(objFolder.decode('utf-8'))
                    # print(dataPath, objFolder)
                    if os.path.isdir(os.path.join(dataPath, objFolder)):
                        objInfoTXT = os.path.join(dataPath, objFolder, 'objInfo.txt')
                        # print(objInfoTXT)
                        with open(objInfoTXT) as objInfoPointer:
                            objInfo = objInfoPointer.readline()
                        objSelectedList.append(objInfo)
                if len(objSelectedList) > 0:
                    if len(outputImages) + len(objSelectedList) > batchSizeof3DShape and len(inputImages)>0:
                        break
                    image2DPath = os.path.join(self._Pascal3DDataPath, objSelectedList[0].split(' ')[1])
                    inputImage = cv2.imread(image2DPath, cv2.IMREAD_COLOR)

                    imageRow, imageCol, _ = inputImage.shape
                    inputImage = cv2.resize(inputImage, imageSize)
                    isFlip = False
                    if self._isTrain:
                        # if np.random.rand()<0.5:
                        #     inputImage = imgAug(inputImage=inputImage,
                        #                         gaussianBlur=True, channelInvert=True, brightness=True, hueSat=True)
                        isFlip = np.random.rand() < 0.5
                        if isFlip:
                            inputImage = cv2.flip(inputImage, flipCode=1)
                    inputImage = (inputImage-self._imagePixelMean)/self._imagePixelStd

                    objOrderingImage = -1 * np.ones([self._gridSize[1], self._gridSize[0], self._predNumPerGrid])
                    bboxHWImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 2])
                    bboxXYImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 2])
                    objnessImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                    EulerRadImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
                    outputPerImage, categoryPerImage, instPerImage, EulerPerImage = [], [], [], []
                    itemIndex = 0
                    for objIndex, objSelected in enumerate(objSelectedList):
                        className,imagePath,CADModelPath,colMin,rowMin,colMax,rowMax,azimuth,elevation,inPlaneRot=objSelected.split(' ')
                        # #border for square
                        # rowMin, rowMax = float(rowMin) + heightBorderSize, float(rowMax) + heightBorderSize
                        # colMin, colMax = float(colMin) + widthBorderSize, float(colMax) + widthBorderSize
                        # augmentation
                        colMin, colMax, rowMin, rowMax = float(colMin), float(colMax), float(rowMin), float(rowMax)
                        if isFlip:
                            colMin, colMax = imageCol - colMax, imageCol - colMin

                        # color = (0, 255, 0)
                        # thickness = 2
                        # p0 = (int(colMin), int(rowMin))
                        # p1 = (int(colMax), int(rowMax))
                        # # print(p0)
                        # # print(p1)
                        # # print(image_bbox2D.shape)
                        # cv2.rectangle(img=inputImage, pt1=p0, pt2=p1, color=color, thickness=thickness)
                        # # cv2.circle(image, (colCenter, rowCenter), 5, (0,0,255), -1)

                        h = (rowMax-rowMin)/imageRow
                        w = (colMax-colMin)/imageCol
                        # colMin = np.max((0.0, colMin))
                        # colMax = np.min((colMax, imageCol))
                        # rowMin = np.max((0.0, rowMin))
                        # rowMax = np.min((rowMax, imageRow))
                        azimuth,elevation,inPlaneRot = float(azimuth)/180.0*np.pi,float(elevation)/180.0*np.pi,float(inPlaneRot)/180.0*np.pi
                        cadIndex = int(CADModelPath.split('/')[-1])

                        rowCenterOnGrid = (rowMax+rowMin)/2.0*self._gridSize[1]/imageRow
                        colCenterOnGrid = (colMax+colMin)/2.0*self._gridSize[0]/imageCol
                        rowIndexOnGrid = int(rowCenterOnGrid)
                        colIndexOnGrid = int(colCenterOnGrid)
                        dx,dy = colCenterOnGrid - colIndexOnGrid, rowCenterOnGrid - rowIndexOnGrid
                        bboxHeight,bboxWidth = np.min((1.0, (rowMax-rowMin)/imageRow)),np.min((1.0, (colMax-colMin)/imageCol))
                        dist_iou = self._dist_IoU(X=np.array([h,w]), centres=self._bboxHWMean)
                        predIndex = np.argmin(dist_iou)
                        is_assigned = False
                        if (rowIndexOnGrid>=0 and rowIndexOnGrid<self._gridSize[1]) \
                            and (colIndexOnGrid>=0 and colIndexOnGrid<self._gridSize[0]) \
                            and bboxHeight > 0 and bboxWidth > 0:
                            while is_assigned == False and predIndex < self._predNumPerGrid:
                                if objnessImage[rowIndexOnGrid, colIndexOnGrid, predIndex] == 0:
                                    # objectness and bounding box
                                    objnessImage[rowIndexOnGrid,colIndexOnGrid,predIndex]=1
                                    bboxHWImage[rowIndexOnGrid,colIndexOnGrid,predIndex,:] = bboxHeight, bboxWidth
                                    bboxXYImage[rowIndexOnGrid,colIndexOnGrid,predIndex,:] = dx, dy
                                    EulerRadImage[rowIndexOnGrid,colIndexOnGrid,predIndex,:] = azimuth,elevation,inPlaneRot
                                    # category vector
                                    categoryVector = np.zeros(len(self._classDict))
                                    categoryVector[self._classDict[className]-1] = 1
                                    # car instance vector
                                    instanceVector = np.zeros(self._instNumMax)
                                    instanceVector[self._instDict[className][CADModelPath]-1] = 1
                                    # object 3d shape
                                    object3DCAD = np.load(os.path.join(self._Pascal3DDataPath, CADModelPath+'.npy'))
                                    object3DCAD = np.where(object3DCAD > 0, 1.0, 0.0)
                                    # Euler angle in rad
                                    EulerRad = np.array([azimuth,elevation,inPlaneRot])

                                    # append items
                                    outputPerImage.append(object3DCAD)
                                    categoryPerImage.append(categoryVector)
                                    instPerImage.append(instanceVector)
                                    EulerPerImage.append(EulerRad)
                                    # set item order
                                    objOrderingImage[rowIndexOnGrid, colIndexOnGrid, predIndex] = itemIndex
                                    itemIndex += 1
                                    is_assigned = True
                                else:
                                    predIndex += 1
                    if itemIndex > 0:
                        # inputImage = cv2.resize(inputImage, imageSize)
                        inputImages.append(inputImage)
                        bboxHWImages.append(bboxHWImage)
                        bboxXYImages.append(bboxXYImage)
                        objnessImages.append(objnessImage)
                        EulerRadImages.append(EulerRadImage)

                        for gridRow in range(self._gridSize[1]):
                            for gridCol in range(self._gridSize[0]):
                                for predIndex in range(self._predNumPerGrid):
                                    objOrder = int(objOrderingImage[gridRow, gridCol, predIndex])
                                    if objOrder>=0:
                                        outputImages.append(outputPerImage[objOrder])
                                        categoryList.append(categoryPerImage[objOrder])
                                        instList.append(instPerImage[objOrder])
                                        EulerRadList.append(EulerPerImage[objOrder])
                self.dataStart += 1
                if self.dataStart >= self.dataLength:
                    self.epoch += 1
                    self._dataPathShuffle()
                    break
        inputImages = np.array(inputImages).astype('float32')
        bboxHWImages = np.array(bboxHWImages).astype('float32')
        bboxXYImages = np.array(bboxXYImages).astype('float32')
        objnessImages = np.array(objnessImages).astype('float32')
        outputImages = np.array(outputImages).astype('float32')
        EulerRadImages = np.array(EulerRadImages).astype('float32')
        categoryList = np.array(categoryList).astype('float32')
        instList = np.array(instList).astype('float32')
        EulerRadList = np.array(EulerRadList).astype('float32')
        offsetX,offsetY = self._getOffset(batchSize=len(inputImages))
        offsetX, offsetY = offsetX.astype('float32'), offsetY.astype('float32')

        # print inputImages.shape
        # print(instList.shape)
        if self._isTrain:
            return offsetX, offsetY, inputImages, objnessImages,\
        bboxHWImages, bboxXYImages, np.sin(EulerRadImages), np.cos(EulerRadImages), \
        outputImages, categoryList, instList
        else:
            return offsetX, offsetY, inputImages, objnessImages, \
                   bboxHWImages, bboxXYImages, np.sin(EulerRadImages), np.cos(EulerRadImages), \
                   outputImages, categoryList, instList, self._imagePixelMean, self._imagePixelStd


# a = Pascal3DMultiObject(Pascal3DDataPath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/')
# a.getImagePrior()
# a.getKMeansBBoxIoU(k=5)





















