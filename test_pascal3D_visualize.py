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
import pangolin
import OpenGL.GL as gl

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

    image_size_network = (448, 448)
    image_size_visualize = (640, 480)

    # prepare pangolin visualizer
    pangolin.CreateWindowAndBind('Main', image_size_visualize[0], image_size_visualize[1])
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    proj = pangolin.ProjectionMatrix(image_size_visualize[0], image_size_visualize[1], 420, 420, 320, 240, 0.1, 1000)
    scam1 = pangolin.OpenGlRenderState(
        proj, pangolin.ModelViewLookAt(0, -2, 0, 0, 0, 0, pangolin.AxisDirection.AxisZ))
    scam2 = pangolin.OpenGlRenderState(
        proj, pangolin.ModelViewLookAt(0, -2, 0, 0, 0, 0, pangolin.AxisDirection.AxisZ))
    scam3 = pangolin.OpenGlRenderState(
        proj, pangolin.ModelViewLookAt(0, -2, 0, 0, 0, 0, pangolin.AxisDirection.AxisZ))
    # scam = pangolin.OpenGlRenderState(
    #     proj, pangolin.ModelViewLookAt(0, -100, -50, 0, 0, 10.1, pangolin.AxisDirection.AxisNegY))

    # Create Interactive View in window
    dcam1 = pangolin.Display('cam1')
    # dcam1.SetBounds(0.0, 1.0, 0.0, 1.0, - 1.0 * image_size_visualize[0] / image_size_visualize[1])  # bound yx start-end
    dcam1.SetAspect(1.0 * image_size_visualize[0] / image_size_visualize[1])
    dcam1.SetHandler(pangolin.Handler3D(scam1))
    # Create Interactive View in window
    dcam2 = pangolin.Display('cam2')
    # dcam2.SetBounds(0.0, 1.0, 0.0, 1.0, - 1.0 * image_size_visualize[0] / image_size_visualize[1])  # bound yx start-end
    dcam2.SetAspect(1.0 * image_size_visualize[0] / image_size_visualize[1])
    dcam2.SetHandler(pangolin.Handler3D(scam2))
    # Create Interactive View in window
    dcam3 = pangolin.Display('cam3')
    # dcam3.SetBounds(0.0, 1.0, 0.0, 1.0, - 1.0 * image_size_visualize[0] / image_size_visualize[1])  # bound yx start-end
    dcam3.SetAspect(1.0 * image_size_visualize[0] / image_size_visualize[1])
    dcam3.SetHandler(pangolin.Handler3D(scam3))

    dImg = pangolin.Display('img1')
    # dImg.SetBounds(0.0, 1.0, 0.0, 1.0, 1.0 * image_size_visualize[0] / image_size_visualize[1])
    dImg.SetAspect(1.0 * image_size_visualize[0] / image_size_visualize[1])
    # dImg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

    view = pangolin.Display('multi')
    view.SetBounds(0.0, 1.0, 0.0, 1.0)
    view.SetLayout(pangolin.LayoutEqual)
    view.AddDisplay(dImg)
    view.AddDisplay(dcam1)
    view.AddDisplay(dcam2)
    view.AddDisplay(dcam3)
    image_texture = pangolin.GlTexture(image_size_visualize[0], image_size_visualize[1], gl.GL_RGB, False, 0, gl.GL_RGB,
                                       gl.GL_UNSIGNED_BYTE)

    panel = pangolin.CreatePanel('ui')
    panel.SetBounds(440. / image_size_visualize[1], 480. / image_size_visualize[1], 600. / image_size_visualize[0], 640. / image_size_visualize[0])
    next_button = pangolin.VarBool('ui.Button', value=False, toggle=False)

    image_col, image_row, batch_size = 448, 448, 1000
    batch_data = data_loader.getNextBatch(batchSizeof3DShape=batch_size,
                                          imageSize=(image_col, image_row),
                                          gridSize=(int(image_col / 32), int(image_row / 32)))
    images = batch_data[2]
    imageMean, imageStd = batch_data[-2], batch_data[-1]
    item_index = 0
    image_visualize = np.zeros_like((image_row, image_col, 3))
    shape_list = []
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        if pangolin.Pushed(next_button):
            pred = model.getPred(input_image=images[item_index], imageMean=imageMean, imageStd=imageStd,
                                 obj_thresh=0.9, IOU_thresh=0.3,
                                 top_1_pred=False, get_3D_shape=True, is_sampling=False)
            img_pred = pred[0]
            shape_list = pred[-1]
            # print(img_pred.shape)
            image_visualize = np.array(cv2.resize(img_pred, image_size_visualize))
            image_visualize = image_visualize[..., [2,1,0]]
            # image_visualize = cv2.cvtColor(image_visualize, cv2.COLOR_BGR2RGB)
            image_visualize = cv2.flip(image_visualize, 0)
            item_index += 1

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        axis_line_width = 3.
        axis_line_length = 0.1
        point_size = 4

        for i, ds in enumerate(zip([dcam1, dcam2, dcam3], [scam1, scam2, scam3])):
            dcam, scam = ds
            dcam.Activate(scam)
            gl.glLineWidth(axis_line_width)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawLine(axis_line_length * np.array([[0, 0, 0], [1, 0, 0]]))
            # Draw lines
            gl.glLineWidth(axis_line_width)
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawLine(axis_line_length * np.array([[0, 0, 0], [0, 1, 0]]))
            # Draw lines
            gl.glLineWidth(axis_line_width)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawLine(axis_line_length * np.array([[0, 0, 0], [0, 0, 1]]))
            if len(shape_list) > i:
                gl.glPointSize(point_size)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(shape_list[i])

        image_texture.Upload(image_visualize, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        dImg.Activate()
        gl.glColor3f(1.0, 1.0, 1.0)
        image_texture.RenderToViewport()

        pangolin.FinishFrame()


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



