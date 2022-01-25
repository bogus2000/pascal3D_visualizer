import src.net_core.darknet as darknet
import src.net_core.autoencoder3D as ae3D
# import src.net_core.priornet as priornet
import src.visualizer.visualizer_ as visualizer
from src.module.function import *
import cv2

config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'predictor_num':5,
        'bbox2D_dim':4, 'bbox3D_dim':3, 'orientation_dim':3,
        'inst_dim':10, 'z_inst_dim':16,
        'activation' : 'elu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : 5*(1+4+3+(2*3+3)+2*16),
        'filter_num_list':[1024,1024,1024,1024],
        'filter_size_list':[3,3,3,1],
        'activation':'elu',
    },
    'decoder':{
        'name':'docoder',
        'input_dim' : 16,
        'output_shape':[64,64,64,1],
        'filter_num_list':[512,256,128,64,1],
        'filter_size_list':[4,4,4,4,4],
        'strides_list':[1,2,2,2,2],
        'activation':'elu',
        'final_activation':'sigmoid'
    },
    # 'prior' : {
    #     'name' : 'priornet',
    #     'input_dim' : 10,  # class num (one-hot vector)
    #     'unit_num_list' : [64, 32, 16],
    #     'core_activation' : 'elu',
    #     'const_log_var' : 0.0,
    # }
}

class nolbo_test(object):
    def __init__(self,
                 nolbo_structure,
                 anchor_boxes=None,
                 ):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']
        self._anchor_boxes = anchor_boxes
        self._buildModel()

    def _buildModel(self):
        print('build Models...')
        self._encoder_core = darknet.Darknet19(name=self._enc_backbone_str['name'], activation='lrelu')
        self._encoder_core_head = darknet.Darknet19_head2D(name=self._enc_backbone_str['name'] + '_head',
                                                           activation='lrelu')
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=[None, None, 1024],
                                            output_dim=self._enc_head_str['output_dim'],
                                            last_pooling=None, activation=self._enc_head_str['activation'])
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        print('done')

    def getPred(self,
                input_image, imageMean, imageStd,
                obj_thresh=0.5, IOU_thresh=0.5,
                top_1_pred=True, get_3D_shape=True, is_sampling=True,
                image_reduced=32):
        # t = time.time()
        if input_image.shape[-1] !=3:
            # if gray image
            input_image = np.stack([input_image,input_image,input_image], axis=-1)
        if input_image.ndim != 4:
            input_image = np.stack([input_image], axis=0)
        _, inputImgRow, inputImgCol, _ = input_image.shape
        self._gridSize = [int(inputImgCol/image_reduced), int(inputImgRow/image_reduced)]

        grid_col, grid_row, predictor_num = self._gridSize[0], self._gridSize[1], self._enc_backbone_str['predictor_num']

        self._enc_output = self._encoder_head(
            self._encoder_core_head(
                self._encoder_core(input_image, training=False)
                , training=False)
            , training=False)
        self._encOutPartitioning()
        # print(1, time.time()-t)
        # t = time.time()

        enc_output_selected = []
        bbox2D_list = []
        enc_output = np.reshape(np.array(self._enc_output), (1, grid_row, grid_col, self._enc_backbone_str['predictor_num'], -1))
        # print(enc_output.shape)
        self._objness = np.array(self._objness)
        self._bboxHW2D = np.array(self._bboxHW2D)
        self._bboxXY2D = np.array(self._bboxXY2D)
        for gr in range(grid_row):
            for gc in range(grid_col):
                objness_pr_list = np.argsort(-self._objness[0, gr, gc, :, 0])
                for prn in objness_pr_list:
                    obj_pr = self._objness[0, gr, gc, prn, 0]
                    if obj_pr > obj_thresh:
                        b2h, b2w = self._bboxHW2D[0, gr, gc, prn, :]
                        b2x, b2y = self._bboxXY2D[0, gr, gc, prn, :]
                        row_min = (float(gr + b2y) / float(grid_row) - b2h / 2.0)
                        row_max = (float(gr + b2y) / float(grid_row) + b2h / 2.0)
                        col_min = (float(gc + b2x) / float(grid_col) - b2w / 2.0)
                        col_max = (float(gc + b2x) / float(grid_col) + b2w / 2.0)
                        bbox2D_list.append([col_min, row_min, col_max, row_max, obj_pr])
                        enc_output_selected.append(enc_output[0, gr, gc, prn, :])
                        if top_1_pred:
                            break
        bbox2D_list = np.array(bbox2D_list)
        enc_output_selected = np.array(enc_output_selected)
        idx_start, idx_end = 1, 1+self._enc_backbone_str['bboxHW2D_dim']+self._enc_backbone_str['bboxXY2D_dim']
        idx_start, idx_end = idx_end, idx_end + self._enc_backbone_str['latent_dim']
        inst_mean_list = enc_output_selected[..., idx_start:idx_end]
        idx_start, idx_end = idx_end, idx_end + self._enc_backbone_str['latent_dim']
        inst_log_var_list = enc_output_selected[..., idx_start:idx_end]
        idx_start, idx_end = idx_end, idx_end + self._enc_backbone_str['orientation_dim']
        sin_mean_list = np.tanh(enc_output_selected[..., idx_start:idx_end])
        idx_start, idx_end = idx_end, idx_end + self._enc_backbone_str['orientation_dim']
        cos_mean_list = np.tanh(enc_output_selected[..., idx_start:idx_end])
        idx_start, idx_end = idx_end, idx_end + self._enc_backbone_str['orientation_dim']
        rad_log_var_list = enc_output_selected[..., idx_start:idx_end]
        # print(2, time.time() - t)
        # t = time.time()


        # bbox2D_list, bbox3D_list = [], []
        # inst_mean_list, inst_log_var_list = [], []
        # sin_mean_list, cos_mean_list, rad_log_var_list = [], [], []
        # self._objness = np.array(self._objness)
        # self._bbox2D = np.array(self._bbox2D)
        # self._bbox3D = np.array(self._bbox3D)
        # self._ori_sin_mean = np.array(self._ori_sin_mean)
        # self._ori_cos_mean = np.array(self._ori_cos_mean)
        # self._rad_log_var = np.array(self._rad_log_var)
        # for gr in range(grid_row):
        #     for gc in range(grid_col):
        #         objness_pr_list = np.argsort(-self._objness[0, gr, gc, :, 0])
        #         for prn in objness_pr_list:
        #             obj_pr = self._objness[0, gr, gc, prn, 0]
        #             if obj_pr > obj_thresh:
        #                 b2h, b2w, b2x, b2y = self._bbox2D[0, gr, gc, prn, :]
        #                 row_min = (float(gr + b2y) / float(grid_row) - b2h / 2.0)
        #                 row_max = (float(gr + b2y) / float(grid_row) + b2h / 2.0)
        #                 col_min = (float(gc + b2x) / float(grid_col) - b2w / 2.0)
        #                 col_max = (float(gc + b2x) / float(grid_col) + b2w / 2.0)
        #                 b3w, b3h, b3l = self._bbox3D[0, gr, gc, prn, :] #hwl
        #                 inst_mean = self._inst_mean[0, gr, gc, prn, :]
        #                 inst_log_var = self._inst_log_var[0, gr, gc, prn, :]
        #                 sin_mean = self._ori_sin_mean[0, gr, gc, prn, :]
        #                 cos_mean = self._ori_cos_mean[0, gr, gc, prn, :]
        #                 rad_log_var = self._rad_log_var[0, gr, gc, prn, :]
        #
        #                 bbox2D_list.append([col_min, row_min, col_max, row_max, obj_pr])
        #                 bbox3D_list.append([b3h, b3w, b3l])
        #                 inst_mean_list.append(inst_mean)
        #                 inst_log_var_list.append(inst_log_var)
        #                 sin_mean_list.append(sin_mean)
        #                 cos_mean_list.append(cos_mean)
        #                 rad_log_var_list.append(rad_log_var)
        #                 if top_1_pred:
        #                     break
        # bbox2D_list = np.array(bbox2D_list)
        # bbox3D_list = np.array(bbox3D_list)
        # inst_mean_list = np.array(inst_mean_list)
        # inst_log_var_list = np.array(inst_log_var_list)
        # sin_mean_list = np.array(sin_mean_list)
        # cos_mean_list = np.array(cos_mean_list)
        # rad_log_var_list = np.array(rad_log_var_list)
        # print(2, time.time()-t)
        # t = time.time()

        if len(bbox2D_list) > 0:
            selected_box_indices = nonMaximumSuppresion(bbox2D_list, IOU_thresh)
        else:
            selected_box_indices = []
        bbox2D_selected = bbox2D_list[selected_box_indices]
        inst_mean_selected = inst_mean_list[selected_box_indices]
        inst_log_var_selected = inst_log_var_list[selected_box_indices]
        sin_mean_selected = sin_mean_list[selected_box_indices]
        cos_mean_selected = cos_mean_list[selected_box_indices]
        rad_log_var_selected = rad_log_var_list[selected_box_indices]
        # print(3, time.time() - t)
        # t = time.time()

        image_bbox2D = input_image[0].copy() * imageStd + imageMean
        imrow, imcol, _ = image_bbox2D.shape
        for bbox2D in bbox2D_selected:
            color = (0, 255, 0)
            thickness = 2
            p0 = (int(bbox2D[0] * imcol), int(bbox2D[1] * imrow))
            p1 = (int(bbox2D[2] * imcol), int(bbox2D[3] * imrow))
            # print(p0)
            # print(p1)
            # print(image_bbox2D.shape)
            cv2.rectangle(img=image_bbox2D, pt1=p0, pt2=p1, color=color, thickness=thickness)

        if get_3D_shape:
            outputs_3D_shape = []
            if len(inst_mean_selected) > 0:
                if is_sampling:
                    sampling_num = 32
                    for inst_mean, inst_log_var in zip(inst_mean_selected, inst_log_var_selected):
                        inst_mean_samples = tf.stack([inst_mean] * sampling_num)
                        inst_log_var_samples = tf.stack([inst_log_var] * sampling_num)
                        latents = sampling(inst_mean_samples, inst_log_var_samples)
                        outputs = tf.reshape(tf.reduce_mean(self._decoder(latents, training=False), axis=0), (64,64,64))
                        outputs_3D_shape.append(outputs)
                    outputs_3D_shape = np.array(outputs_3D_shape)
                else:
                    voxels_predicted = np.array(tf.reshape(self._decoder(inst_mean_selected), (-1, 64,64,64)))
                    for voxel in voxels_predicted:
                        voxel = visualizer.objRescaleTransform(voxel)
                        outputs_3D_shape.append(voxel)

            return image_bbox2D, bbox2D_selected, \
            sin_mean_selected, cos_mean_selected, rad_log_var_selected,\
            outputs_3D_shape
        else:
            return image_bbox2D, bbox2D_selected, \
            sin_mean_selected, cos_mean_selected, rad_log_var_selected

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_backbone_str['name']
        self._encoder_core.load_weights(os.path.join(load_path, file_name))
    def loadEncoderHead(self, load_path):
        file_name = self._enc_backbone_str['name'] + '_head'
        self._encoder_core_head.load_weights(os.path.join(load_path, file_name))
        file_name = self._enc_head_str['name']
        self._encoder_head.load_weights(os.path.join(load_path, file_name))
    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)
    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)

    def _encOutPartitioning(self):
        pr_num = self._enc_backbone_str['predictor_num']
        self._objness, self._bboxHW2D, self._bboxXY2D = [], [], []
        self._latent_mean, self._latent_log_var = [], []
        self._ori_sin_mean, self._ori_cos_mean, self._rad_log_var = [], [], []
        part_start = 0
        part_end = part_start
        for predIndex in range(pr_num):
            # objectness
            part_end += 1
            self._objness.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bboxHW2D_dim']
            self._bboxHW2D.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bboxXY2D_dim']
            self._bboxXY2D.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['latent_dim']
            self._latent_mean.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['latent_dim']
            self._latent_log_var.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._ori_sin_mean.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._ori_cos_mean.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._rad_log_var.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
        self._objness = tf.sigmoid(tf.transpose(tf.stack(self._objness), [1, 2, 3, 0, 4]))
        self._bboxHW2D = tf.exp(tf.transpose(tf.stack(self._bboxHW2D), [1, 2, 3, 0, 4])) * self._anchor_boxes
        self._bboxXY2D = tf.sigmoid(tf.transpose(tf.stack(self._bboxXY2D), [1, 2, 3, 0, 4]))
        self._latent_mean = tf.transpose(tf.stack(self._latent_mean), [1, 2, 3, 0, 4])
        self._latent_log_var = tf.clip_by_value(tf.transpose(tf.stack(self._latent_log_var), [1, 2, 3, 0, 4]),
                                                clip_value_min=-10.0, clip_value_max=10.0)
        self._ori_sin_mean = tf.tanh(tf.transpose(tf.stack(self._ori_sin_mean), [1, 2, 3, 0, 4]))
        self._ori_cos_mean = tf.tanh(tf.transpose(tf.stack(self._ori_cos_mean), [1, 2, 3, 0, 4]))
        self._rad_log_var = tf.clip_by_value(tf.transpose(tf.stack(self._rad_log_var), [1, 2, 3, 0, 4]),
                                             clip_value_min=-3.0, clip_value_max=3.0)



