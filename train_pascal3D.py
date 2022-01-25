import numpy as np
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 0 = all messages are logged(default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages  arenot printed
# 3 = INFO, WARNING, and ERROR messages  arenot printed

import src.dataset_loader.pascal3D as pascal3D
import src.module.nolbo as nolbo
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
    # [416,416, 24],
    [448,448, 24],
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
        trainOrVal='train',
        Pascal3DDataPath='/home/yonsei/dataset/PASCAL3D+_release1.1/',
        isTrain=True,
        # Pascal3DDataPath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/'
    )
    anchor_boxes = np.load('./src/dataset_loader/bboxMean.npy').astype('float32')
    model = nolbo.nolbo(nolbo_structure=config,
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

    loss = np.zeros(14)
    epoch, iteration, run_time = 0., 0., 0.
    imgSizeAndBatchList = imageSizeAndBatchListPascal

    print('start training...')
    while epoch < training_epoch:
        start_time = time.time()
        periodOfImageSize = 3
        if int(iteration) % (periodOfImageSize * len(imgSizeAndBatchList)) == 0:
            np.random.shuffle(imgSizeAndBatchList)
        image_col, image_row, batch_size = imgSizeAndBatchList[int(iteration) % int((periodOfImageSize * len(imgSizeAndBatchList)) / periodOfImageSize)]
        batch_data = data_loader.getNextBatch(batchSizeof3DShape=batch_size,
                                              imageSize=(image_col, image_row),
                                              gridSize=(int(image_col/32), int(image_row/32)))
        epoch_curr = data_loader.epoch
        data_start = data_loader.dataStart
        data_length = data_loader.dataLength

        if epoch_curr != epoch:
            print('')
            iteration = 0
            loss = loss * 0.
            run_time = 0.
            if save_path != None:
                print('save model...')
                model.saveModel(save_path=save_path)
        epoch = epoch_curr
        # try:
        loss_temp = model.fit(inputs=batch_data)

        end_time = time.time()

        loss = (loss * iteration + np.array(loss_temp)) / (iteration + 1.0)
        run_time = (run_time * iteration + (end_time - start_time)) / (iteration + 1.0)
        sys.stdout.write(
            "Ep:{:03d} it:{:04d} rt:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("cur/tot:{:04d}/{:04d} b:({},{}) ".format(data_start, data_length, len(batch_data[0]),
                                                                   len(batch_data[-1])))
        sys.stdout.write(
            "obj:{:.2f}, noobj:{:.2f}, b2D:{:.3f}, ".format(loss[0], loss[1], loss[2]))
        sys.stdout.write(
            "b2DCIOU:{:.3f}, Shape:{:.2f}, KL:{:.2f}, reg:{:.2f} ".format(loss[3], loss[4], loss[5], loss[6]))
        sys.stdout.write(
            "scMSE:{:.2f}, sc1:{:.2f}, scKL:{:.2f} ".format(loss[7], loss[8], loss[9]))
        sys.stdout.write(
            "op:{:.3f}, np:{:.3f} ".format(loss[10], loss[11]))
        sys.stdout.write(
            "pr:{:.4f}, rc:{:.4f}   \r".format(loss[12], loss[13]))
        sys.stdout.flush()

        if np.sum(loss) != np.sum(loss):
            print('')
            print('NaN')
            return
        iteration += 1.0

        # except:
        #     print('save model...')
        #     model.saveModel(save_path=save_path)
        #     print(image_col, image_row, len(batch_data[-1]), len(batch_data[0]))
        #     return

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
    'prior_category' : {
        'name' : 'priornet_category',
        'input_dim' : category_num,  # class num (one-hot vector)
        'unit_num_list' : [latent_dim//4, latent_dim//2],
        'core_activation' : 'relu',
        'const_log_var' : 0.0,
    },
    'prior_instance' : {
        'name' : 'priornet_instance',
        'input_dim' : category_num + inst_num,  # class num (one-hot vector)
        'unit_num_list' : [latent_dim//4, latent_dim//2],
        'core_activation' : 'relu',
        'const_log_var' : 0.0,
    }
}

if __name__ == '__main__':
    sys.exit(train(
        training_epoch=20, learning_rate=1e-4,
        config=config,
        save_path='./weights/pascal3D/',
        load_path='./weights/pascal3D/',
        # load_encoder_backbone_path='./weights/yolov2/',
        # load_encoder_backbone_name='nolbo_backbone',
        # load_decoder_path='./weights/AE3D/',
        # load_decoder_name='decoder3D',
    ))



