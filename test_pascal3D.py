import cv2
import numpy as np
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 0 = all messages are logged(default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages  arenot printed
# 3 = INFO, WARNING, and ERROR messages  arenot printed

import src.dataset_loader.pascal3D as pascal3D
import src.module.nolbo_test as nolbo_test
import tensorflow as tf

tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

imageSizeAndBatchListKITTI = [
    # # [384,128,64],
    # # [480,160,64],
    # [576,192,64,26],
    # [672,224,64,24],
    # [768,256,64,22],
    [864,288,58,20],
    [960,288,52,18],
    [960,320,50,16],
    [960,352,48,14],
    [1056,352,46,12],
    [1088,352,46,12],
    [1088,384,44,10],
    [1152,384,42,10],
    [1216,384,42,8],
    [1216,448,42,8],
    [1248,384,42,8],
    [1248,448,42,8],
    # # #############################
    # [160,128,64],
    # [288,288,56],
    # [320,256,48],
    # [320,320,48],
    # [416,416,40],
    # [480,384,40],
    # [448,448,40],
    # [640,512,32],
    # [800,640,24],
]

imageSizeAndBatchListPascal = [
    # [224, 92],
    # [256,256, 96],
    # [288,288, 90],
    # [320,320, 82],
    # [352, 352, 32],
    # [384,384, 62],
    [416,416, 24],
    # [448,448, 50],
    # [480, 50],
    # [512, 30],
    # [544, 25],
    # [576, 24],
    # [608, 22],
]

def train(
        training_epoch = 1000,
        learning_rate = 1e-4,
        config = None,
        save_path = None, load_path = None,
        load_encoder_backbone_path = None, load_encoder_backbone_name = None,
        load_decoder_path = None, load_decoder_name = None,
):
    data_loader = pascal3D.Pascal3DMultiObject(
        imageSize=(640, 480),
        gridSize=(20, 15),
        predNumPerGrid=5,
        trainOrVal='val',
        Pascal3DDataPath='/home/yonsei/dataset/PASCAL3D+_release1.1/',
        isTrain=False,
        # Pascal3DDataPath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/',
    )
    anchor_boxes = np.load('./src/dataset_loader/bboxMean.npy').astype('float32')
    model = nolbo_test.nolbo_test(nolbo_structure=config,
                        anchor_boxes=anchor_boxes)

    if load_path != None:
        print('load weights...')
        model.loadModel(load_path=load_path)
        # model.loadEncoder(load_path=load_path)
        # model.loadEncoderBackbone(load_path=load_path)
        # model.loadEncoderHead(load_path=load_path)
        # model.loadDecoder(load_path=load_path)
        # model.loadPriornet(load_path=load_path)
        print('done!')

    if load_encoder_backbone_path != None:
        print('load encoder backbone weights...')
        model.loadEncoderBackbone(
            load_path=load_encoder_backbone_path,
            file_name=load_encoder_backbone_name
        )
        print('done!')

    if load_decoder_path != None:
        print('load decoder weights...')
        model.loadDecoder(
            load_path=load_decoder_path,
            file_name=load_decoder_name
        )
        print('done!')

    epoch, iteration, run_time = 0., 0., 0.

    item_num = 0
    print('start training...')
    while epoch < training_epoch:
        image_col, image_row, batch_size = 448, 448, 24
        batch_data = data_loader.getNextBatch(batchSizeof3DShape=batch_size,
                                              imageSize=(image_col, image_row),
                                              gridSize=(int(image_col/32), int(image_row/32)))
        images = batch_data[2]
        imageMean, imageStd = batch_data[-2], batch_data[-1]
        for img in images:
            pred = model.getPred(input_image=img, imageMean=imageMean, imageStd=imageStd,
                                 obj_thresh=0.9, IOU_thresh=0.3,
                                 top_1_pred=False, get_3D_shape=False, is_sampling=False)
            img_pred = pred[0]
            cv2.imwrite(os.path.join('./data/test/img/', '{:03d}.png'.format(item_num)), img_pred)
            item_num += 1

predictor_num = 5
latent_dim = 64
category_num = 12
inst_num = 10
config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'predictor_num': predictor_num,
        'bboxHW2D_dim':2, 'bboxXY2D_dim':2, 'orientation_dim':3,
        'latent_dim':latent_dim,
        'z_category_dim' : latent_dim//2,
        'z_instance_dim' : latent_dim//2,
        'activation' : 'lrelu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : predictor_num*(1+4+(2*3+3)+2*latent_dim),  # objness + hwxy + whl + (2*sincosAEI+radAEI) + 2*latent
        'activation':'relu',
    },
    'decoder':{
        'name':'decoder',
        'input_dim' : latent_dim,
        'output_shape':[64,64,64,1],
        'filter_num_list':[512,256,128,64,1],
        'filter_size_list':[4,4,4,4,4],
        'strides_list':[1,2,2,2,2],
        'activation':'relu',
        'final_activation':'sigmoid'
    },
}

if __name__ == '__main__':
    sys.exit(train(
        training_epoch=1000, learning_rate=1e-4,
        config=config,
        save_path='./weights/pascal3D/',
        load_path='./weights/pascal3D/',
        # load_encoder_backbone_path='./weights/yolov2/',
        # load_encoder_backbone_name='nolbo_backbone',
        # load_decoder_path='./weights/AE3D/',
        # load_decoder_name='decoder3D',
    ))



