import src.net_core.darknet as darknet
import src.net_core.autoencoder3D as ae3D
import src.net_core.priornet as priornet
import numpy as np

from src.module.function import *

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
        'filter_num_list':[1024,1024,1024],
        'filter_size_list':[3,3,3],
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
    'prior' : {
        'name' : 'priornet',
        'input_dim' : 10,  # class num (one-hot vector)
        'unit_num_list' : [64, 32, 16],
        'core_activation' : 'elu',
        'const_log_var' : 0.0,
    }
}

class nolbo(object):
    def __init__(self, nolbo_structure,
                 learning_rate=1e-4,
                 anchor_boxes=None):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        # self._name = nolbo_structure['name']
        # self._predictor_num = nolbo_structure['predictor_num']
        # self._bbox2D_dim = nolbo_structure['bbox2D_dim']
        # self._bbox3D_dim = nolbo_structure['bbox3D_dim']
        # self._orientation_dim = nolbo_structure['orientation_dim']
        # self._inst_dim = nolbo_structure['inst_dim']
        # self._z_inst_dim = nolbo_structure['z_inst_dim']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']
        self._prior_category_str = nolbo_structure['prior_category']
        self._prior_instance_str = nolbo_structure['prior_instance']

        self._rad_var = (15.0/180.0 * 3.141593) ** 2

        self._anchor_boxes = anchor_boxes

        # # self._strategy = strategy
        # self._strategy = tf.distribute.MirroredStrategy()
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        self._buildModel()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _buildModel(self):
        print('build Models...')
        self._encoder_core = darknet.Darknet19(name=self._enc_backbone_str['name'], activation='lrelu')
        self._encoder_core_head = darknet.Darknet19_head2D(name=self._enc_backbone_str['name']+'_head', activation='lrelu')
        #==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=[None,None,1024],
                                            output_dim=self._enc_head_str['output_dim'],
                                            last_pooling=None, activation=self._enc_head_str['activation'])
        #==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        self._priornet_category = priornet.priornet(structure=self._prior_category_str)
        self._priornet_instance = priornet.priornet(structure=self._prior_instance_str)
        print('done')

    def fit(self, inputs):
        self._getInputs(inputs=inputs)
        with tf.GradientTape() as tape:
            # get encoder output and loss
            self._enc_output = self._encoder_head(
                self._encoder_core_head(
                    self._encoder_core(self._input_images, training=True)
                    , training=True)
                , training=True)

            self._getEncoderOutputs()

            # get (priornet, decoder) output and loss
            self._category_mean_prior, self._category_log_var_prior = self._priornet_category(self._category_vectors_gt, training=True)
            self._inst_mean_prior, self._inst_log_var_prior = self._priornet_instance(self._inst_vectors_gt, training=True)
            self._latent_mean_prior = tf.concat([self._category_mean_prior, self._inst_mean_prior], axis=-1)
            self._latent_log_var_prior = tf.concat([self._category_log_var_prior, self._inst_log_var_prior], axis=-1)

            self._selectObjFromTile()
            self._latents = sampling(mu=self._latent_mean_sel, logVar=self._latent_log_var_sel)
            self._outputs = self._decoder(self._latents, training=True)

            self._getEncoderLoss()
            self._getDecoderAndPriorLoss()

            # # get network parameter regulization loss
            # reg_loss = tf.reduce_sum(self._encoder_head.losses + self._encoder_backbone.losses + self._decoder.losses + self._priornet.losses)
            self._loss_objness = tf.reduce_mean(self._loss_objness, axis=0)
            self._loss_no_objness = tf.reduce_mean(self._loss_no_objness, axis=0)
            self._loss_bbox2D = tf.reduce_mean(self._loss_bbox2D_hw + self._loss_bbox2D_xy, axis=0)
            self._loss_bbox2D_CIOU = tf.reduce_mean(self._loss_bbox2D_CIOU, axis=0)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)
            self._loss_latents_kl = tf.reduce_mean(self._loss_latents_kl, axis=0)
            self._loss_prior_reg = tf.reduce_mean(self._loss_prior_reg, axis=0)
            self._loss_sincos_mse = tf.reduce_mean(self._loss_sincos_mse, axis=0)
            self._loss_sincos_1 = tf.reduce_mean(self._loss_sincos_1, axis=0)
            self._loss_sincos_kl = tf.reduce_mean(self._loss_sincos_kl, axis=0)

            # total loss
            total_loss = (
                    50.0 * self._loss_objness + 0.1 * self._loss_no_objness
                    + 20.0 * self._loss_bbox2D
                    + 20.0 * self._loss_bbox2D_CIOU
                    + self._loss_shape
                    + self._loss_latents_kl
                    + 0.01 * self._loss_prior_reg
                    + (100.0 * self._loss_sincos_mse + 1000.0 * self._loss_sincos_1)
                    + 0.01 * self._loss_sincos_kl
                    # + reg_loss
            )

        trainable_variables = self._encoder_core.trainable_variables\
                              + self._encoder_core_head.trainable_variables + self._encoder_head.trainable_variables \
                              + self._decoder.trainable_variables \
                              + self._priornet_category.trainable_variables + self._priornet_instance.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        # ==== evaluations
        self._objnessEval()
        self._obj_prb = tf.reduce_mean(self._obj_prb)
        self._no_obj_prb = tf.reduce_mean(self._no_obj_prb)

        TP, FP, FN = voxelPrecisionRecall(xTarget=self._output_images_gt, xPred=self._outputs)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10))
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10))

        return self._loss_objness, self._loss_no_objness,\
               self._loss_bbox2D,\
               self._loss_bbox2D_CIOU,\
            self._loss_shape, self._loss_latents_kl,\
            self._loss_prior_reg, \
            self._loss_sincos_mse, self._loss_sincos_1, self._loss_sincos_kl,\
            self._obj_prb, self._no_obj_prb, \
            pr, rc

    def saveEncoderBackbone(self, save_path):
        file_name = self._enc_backbone_str['name']
        self._encoder_core.save_weights(os.path.join(save_path, file_name))
    def saveEncoderHead(self, save_path):
        file_name = self._enc_backbone_str['name']+'_head'
        self._encoder_core_head.save_weights(os.path.join(save_path, file_name))
        file_name = self._enc_head_str['name']
        self._encoder_head.save_weights(os.path.join(save_path, file_name))
    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)
    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))
    def savePriornet(self, save_path):
        file_name = self._prior_category_str['name']
        self._priornet_category.save_weights(os.path.join(save_path, file_name))
        file_name = self._prior_instance_str['name']
        self._priornet_instance.save_weights(os.path.join(save_path, file_name))
    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriornet(save_path=save_path)

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
    def loadPriornet(self, load_path):
        file_name = self._prior_category_str['name']
        self._priornet_category.load_weights(os.path.join(load_path, file_name))
        file_name = self._prior_instance_str['name']
        self._priornet_instance.load_weights(os.path.join(load_path, file_name))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriornet(load_path=load_path)

    def _getInputs(self, inputs):
        self._offset_x, self._offset_y,\
        self._input_images,\
        self._objness_gt, self._bboxHW2D_gt, self._bboxXY2D_gt,\
        self._ori_sin_gt, self._ori_cos_gt, \
        self._output_images_gt, self._category_vectors_gt, self._inst_vectors_gt = inputs

        self._offset_x = tf.convert_to_tensor(self._offset_x)
        self._offset_y = tf.convert_to_tensor(self._offset_y)
        self._input_images = tf.convert_to_tensor(self._input_images)
        self._objness_gt = tf.convert_to_tensor(self._objness_gt)
        self._bboxHW2D_gt = tf.convert_to_tensor(self._bboxHW2D_gt)
        self._bboxXY2D_gt = tf.convert_to_tensor(self._bboxXY2D_gt)
        self._ori_sin_gt = tf.convert_to_tensor(self._ori_sin_gt)
        self._ori_cos_gt = tf.convert_to_tensor(self._ori_cos_gt)
        self._output_images_gt = tf.convert_to_tensor(self._output_images_gt)
        self._category_vectors_gt = tf.convert_to_tensor(self._category_vectors_gt)
        self._inst_vectors_gt = tf.convert_to_tensor(self._inst_vectors_gt)

        self._inst_vectors_gt = tf.concat([self._category_vectors_gt, self._inst_vectors_gt], axis=-1)


    def _getEncoderOutputs(self):
        self._encOutPartitioning()
        self._getbbox2DIOU()

    def _getEncoderLoss(self):
        self._objnessLoss()
        self._bbox2DLoss()
        self._bbox2DLossCIOU()
        self._poseLoss()

    def _getDecoderAndPriorLoss(self):
        self._objLatentAndShapeLoss()
        self._priorRegLoss()

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
        self._latent_log_var = tf.clip_by_value(tf.transpose(tf.stack(self._latent_log_var), [1, 2, 3, 0, 4]), clip_value_min=-10.0, clip_value_max=10.0)
        self._ori_sin_mean = tf.tanh(tf.transpose(tf.stack(self._ori_sin_mean), [1, 2, 3, 0, 4]))
        self._ori_cos_mean = tf.tanh(tf.transpose(tf.stack(self._ori_cos_mean), [1, 2, 3, 0, 4]))
        self._rad_log_var = tf.clip_by_value(tf.transpose(tf.stack(self._rad_log_var), [1, 2, 3, 0, 4]), clip_value_min=-3.0, clip_value_max=3.0)

    def _getbbox2DIOU(self):
        len_grid_x, len_grid_y = tf.cast(tf.shape(self._offset_x)[2], tf.float32), tf.cast(tf.shape(self._offset_x)[1], tf.float32)
        xmin_gt = (self._bboxXY2D_gt[..., 0] + self._offset_x)/len_grid_x - self._bboxHW2D_gt[..., 1]/2.
        xmax_gt = (self._bboxXY2D_gt[..., 0] + self._offset_x)/len_grid_x + self._bboxHW2D_gt[..., 1]/2.
        ymin_gt = (self._bboxXY2D_gt[..., 1] + self._offset_y)/len_grid_y - self._bboxHW2D_gt[..., 0]/2.
        ymax_gt = (self._bboxXY2D_gt[..., 1] + self._offset_y)/len_grid_y + self._bboxHW2D_gt[..., 0]/2.

        xmin = (self._bboxXY2D[..., 0] + self._offset_x)/len_grid_x - self._bboxHW2D[..., 1] / 2.
        xmax = (self._bboxXY2D[..., 0] + self._offset_x)/len_grid_x + self._bboxHW2D[..., 1] / 2.
        ymin = (self._bboxXY2D[..., 1] + self._offset_y)/len_grid_y - self._bboxHW2D[..., 0] / 2.
        ymax = (self._bboxXY2D[..., 1] + self._offset_y)/len_grid_y + self._bboxHW2D[..., 0] / 2.

        xmin_inter = tf.math.maximum(xmin_gt, xmin)
        ymin_inter = tf.math.maximum(ymin_gt, ymin)
        xmax_inter = tf.math.minimum(xmax_gt, xmax)
        ymax_inter = tf.math.minimum(ymax_gt, ymax)
        intersection_xlen = tf.maximum(xmax_inter - xmin_inter, 0.0)
        intersection_ylen = tf.maximum(ymax_inter - ymin_inter, 0.0)
        # print(intersection_xlen)
        # print(self._bbox2D_tile)
        intersection_area = intersection_xlen * intersection_ylen
        box_gt_area = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)
        box_pr_area = (xmax - xmin) * (ymax - ymin)
        union_area = tf.maximum(box_gt_area + box_pr_area - intersection_area, 1e-9)
        self._IOU = tf.clip_by_value(intersection_area/union_area, 0., 1.)

        xmin_out = tf.minimum(xmin_gt, xmin)
        ymin_out = tf.minimum(ymin_gt, ymin)
        xmax_out = tf.maximum(xmax_gt, xmax)
        ymax_out = tf.maximum(ymax_gt, ymax)
        outer_xlen = tf.maximum(xmax_out - xmin_out, 0.)
        outer_ylen = tf.maximum(ymax_out - ymin_out, 0.)
        c2 = tf.square(outer_xlen) + tf.square(outer_ylen)  # sqr of diagonal length of max-outer box
        c2 = tf.maximum(c2, 1e-9)
        box_gt_x = (xmax_gt + xmin_gt) / 2.
        box_gt_y = (ymax_gt + ymin_gt) / 2.
        box_pr_x = (xmax + xmin) / 2.
        box_pr_y = (ymax + ymin) / 2.
        center_diff2 = tf.square(box_gt_x - box_pr_x) + tf.square(box_gt_y - box_pr_y)
        self._RDIOU = center_diff2 / c2

    def _objnessLoss(self):
        d_objness = -self._objness_gt * tf.math.log(self._objness + 1e-10)
        d_no_objness = - (1.0-self._objness_gt) * tf.math.log(1.0-self._objness + 1e-10)
        # d_no_objness = self._ignore_mask * d_no_objness[..., 0]
        d_no_objness = d_no_objness[..., 0]

        self._loss_objness = tf.reduce_sum(d_objness, axis=[1, 2, 3, 4])
        self._loss_no_objness = tf.reduce_sum(d_no_objness, axis=[1, 2, 3])

    def _bbox2DLoss(self):
        # tile shape = (batch, gridy, gridx, 2*predictornum, hwxy)
        square_d_xy = tf.reduce_sum(tf.square(self._bboxXY2D - self._bboxXY2D_gt), axis=-1)
        obj_mask = self._objness_gt[..., 0]
        d_h = obj_mask * (self._bboxHW2D[..., 0] - self._bboxHW2D_gt[..., 0])
        d_w = obj_mask * (self._bboxHW2D[..., 1] - self._bboxHW2D_gt[..., 1])
        self._box_loss_scale = tf.constant((2. - self._bboxHW2D_gt[..., 0] * self._bboxHW2D_gt[..., 1]))
        xy_loss = obj_mask * self._box_loss_scale * square_d_xy
        hw_loss = obj_mask * self._box_loss_scale * (tf.square(d_h) + tf.square(d_w))
        self._loss_bbox2D_xy = tf.reduce_sum(xy_loss, axis=[1, 2, 3])
        self._loss_bbox2D_hw = tf.reduce_sum(hw_loss, axis=[1, 2, 3])

    def _smoothL1(self, x_src, x_trg, cond):
        # return tf.losses.huber(x_src, x_trg, cond)
        return tf.where(tf.abs(x_src - x_trg) > cond, tf.abs(x_src - x_trg) - 0.5 * cond, 0.5 / cond * tf.square(x_src - x_trg))

    def _bbox2DLossCIOU(self):
        obj_mask = self._objness_gt[..., 0]
        pi = 3.14159265358979323846
        # v = ((atan(w/h_gt) - atan(w/h_pr)) / (pi/2) )^2
        # bbox = hwxy
        h_pred = self._bboxHW2D[..., 0]
        w_pred = self._bboxHW2D[..., 1]
        h_gt = self._bboxHW2D_gt[..., 0]
        w_gt = self._bboxHW2D_gt[..., 1]
        ar_gt = w_gt / (h_gt + 1e-9)
        ar = w_pred / (h_pred + 1e-9)
        v = 4. / (pi * pi) * tf.square(tf.atan(ar_gt) - tf.atan(ar))
        alpha = v / (1. - self._IOU + v + 1e-9)
        loss_CIOU = obj_mask * (1. - self._IOU + self._RDIOU + alpha * v)
        # loss_CIOU = obj_mask * (1. - self._IOU)
        bbox_coor_loss = obj_mask * tf.reduce_sum(self._smoothL1(self._bboxHW2D, self._bboxHW2D_gt, 1e-4), axis=-1)
        # bbox_coor_loss = obj_mask * tf.reduce_sum(tf.square(self._bbox2D_dim - self._bbox2D_dim_gt), axis=-1)
        self._loss_bbox2D_CIOU = tf.reduce_sum(loss_CIOU + bbox_coor_loss, axis=[1, 2, 3])
        # loss_IOU = obj_mask * (1. - self._IOU)
        # self._loss_bbox2D_IOU = tf.reduce_sum(loss_IOU, axis=[1,2,3])

    def _getEV(self, sin, cos, radLogVar):
        Esin = tf.exp(-tf.exp(radLogVar) / 2.0) * sin
        Ecos = tf.exp(-tf.exp(radLogVar) / 2.0) * cos
        Varsin = 0.5 - 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (1.0 - 2.0 * sin * sin) - tf.exp(
            -tf.exp(radLogVar)) * sin * sin
        Varcos = 0.5 + 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (2.0 * cos * cos - 1.0) - tf.exp(
            -tf.exp(radLogVar)) * cos * cos
        logVarsin = tf.math.log(Varsin + 1e-7)
        logVarcos = tf.math.log(Varcos + 1e-7)
        return Esin, Ecos, logVarsin, logVarcos

    def _poseLoss(self):
        Esin_gt, Ecos_gt, log_var_sin_gt, log_var_cos_gt = self._getEV(
            sin=self._ori_sin_gt, cos=self._ori_cos_gt, radLogVar=tf.math.log(self._rad_var+1e-9))
        Esin_pr, Ecos_pr, log_var_sin_pr, log_var_cos_pr = self._getEV(
            sin=self._ori_sin_mean, cos=self._ori_cos_mean, radLogVar=self._rad_log_var)

        loss_sin_kl = kl_loss(mean=Esin_pr, logVar=log_var_sin_pr, mean_target=Esin_gt, logVar_target=log_var_sin_gt)
        loss_cos_kl = kl_loss(mean=Ecos_pr, logVar=log_var_cos_pr, mean_target=Ecos_gt, logVar_target=log_var_cos_gt)

        # self._loss_sincos_mse = tf.square(Esin_gt - ori_sinz_tile) / tf.exp(log_var_sin_gt) + tf.square(Ecos_gt - ori_cosz_tile) / tf.exp(log_var_cos_gt)
        self._loss_sincos_mse = tf.square(self._ori_sin_gt - self._ori_sin_mean) + tf.square(self._ori_cos_gt - self._ori_cos_mean) \
                                + tf.square(self._rad_log_var - tf.math.log(self._rad_var+1e-9)) \
                                + tf.square(1. - (self._ori_sin_gt*self._ori_sin_mean + self._ori_cos_gt*self._ori_cos_mean)) \
                                + tf.square(self._ori_sin_gt*self._ori_cos_mean - self._ori_cos_gt*self._ori_sin_mean)
                                # + tf.square(Esin_gt - ori_sinz_tile)/tf.exp(log_var_sin_gt) + tf.square(Ecos_gt - ori_cosz_tile)/tf.exp(log_var_cos_gt) \
        self._loss_sincos_1 = tf.square(tf.square(self._ori_sin_mean)+tf.square(self._ori_cos_mean) - 1.0)

        obj_mask = self._objness_gt[..., 0]
        self._loss_sincos_kl = tf.reduce_sum(obj_mask * (loss_sin_kl + loss_cos_kl), axis=[1, 2, 3])
        self._loss_sincos_mse = tf.reduce_sum(self._objness_gt * self._loss_sincos_mse, axis=[1, 2, 3, 4])
        self._loss_sincos_1 = tf.reduce_sum(self._objness_gt * self._loss_sincos_1, axis=[1, 2, 3, 4])

    def _selectObjFromTile(self):
        obj_mask = tf.cast(self._objness_gt[..., 0], tf.int32)  # (batch, row, col)
        self._latent_mean_sel = tf.dynamic_partition(self._latent_mean, obj_mask, 2)[1]
        self._latent_log_var_sel = tf.dynamic_partition(self._latent_log_var, obj_mask, 2)[1]
        self._ori_sin_mean_sel = tf.dynamic_partition(self._ori_sin_mean, obj_mask, 2)[1]
        self._ori_cos_mean_sel = tf.dynamic_partition(self._ori_cos_mean, obj_mask, 2)[1]
        self._rad_log_var_sel = tf.dynamic_partition(self._rad_log_var, obj_mask, 2)[1]

    def _objLatentAndShapeLoss(self):
        self._loss_latents_kl = kl_loss(mean=self._latent_mean_sel, logVar=self._latent_log_var_sel,
                                        mean_target=self._latent_mean_prior, logVar_target=self._latent_log_var_prior)
        self._loss_shape = binary_loss(xPred=self._outputs, xTarget=self._output_images_gt, gamma=0.60, b_range=False)

    def _priorRegLoss(self):
        self._loss_prior_reg = regulizer_loss(z_mean=self._category_mean_prior,
                                              z_logVar=self._category_log_var_prior,
                                              dist_in_z_space=2.0 * self._enc_backbone_str['z_category_dim'])
        self._loss_prior_reg += regulizer_loss(z_mean=self._inst_mean_prior,
                                               z_logVar=self._inst_log_var_prior,
                                               dist_in_z_space=3.0 * self._enc_backbone_str['z_instance_dim'],
                                               class_input=self._category_vectors_gt)

    def _objnessEval(self):
        self._obj_prb = (
                tf.reduce_sum(self._objness_gt * self._objness, axis=[1, 2, 3, 4])
                / tf.reduce_sum(self._objness_gt, axis=[1, 2, 3, 4])
        )

        self._no_obj_prb = (
                tf.reduce_sum((1.0 - self._objness_gt) * (1.0 - self._objness), axis=[1, 2, 3, 4])
                / tf.reduce_sum(1.0 - self._objness_gt, axis=[1, 2, 3, 4])

        )


class nolbo_single(object):
    def __init__(self, encoder_backbone=None,
                 decoder_structure=None,
                 prior_class_structure=None,
                 prior_inst_structure=None,
                 BATCH_SIZE_PER_REPLICA=32, strategy=None,
                 learning_rate = 1e-4
                 ):
        self._rad_var = (15.0/180.0 * 3.141593) ** 2
        self._dec_str = decoder_structure
        self._prior_cl_str = prior_class_structure
        self._prior_inst_str = prior_inst_structure

        self._strategy = strategy
        # self._strategy = tf.distribute.MirroredStrategy()
        self._GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        with self._strategy.scope():
            self._encoder_backbone = encoder_backbone
            self._buildModel()
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            # self._optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
            # self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def _buildModel(self):
        print('build models....')
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name='nolbo_encoder_head',
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=(2*3+3 + 2*(8+8)),
                                            filter_num_list=[1024, 1024, 1024],
                                            filter_size_list=[3, 3, 3],
                                            last_pooling='max', activation='elu')
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        self._priornet_cl = priornet.priornet(structure=self._prior_cl_str)
        self._priornet_inst = priornet.priornet(structure=self._prior_inst_str)
        print('done')

    def fit(self, inputs):
        class_list, inst_list, sin_gt, cos_gt, input_images, output_images_gt = inputs
        with tf.GradientTape() as tape:
            # get encoder output
            enc_output = self._encoder_head(self._encoder_backbone(input_images, training=True), training=True)
            inst_mean = enc_output[..., :8]
            inst_log_var = enc_output[..., 8:16]
            class_mean = enc_output[..., 16:16+8]
            class_log_var = enc_output[..., 16+8:16+16]
            sin_mean = tf.tanh(enc_output[..., 16+16: 16+16+3])
            cos_mean = tf.tanh(enc_output[..., 16+16+3:16+16+3+3])
            rad_log_var = enc_output[..., 16+16+3+3:]
            mean = tf.concat([inst_mean, class_mean], axis=-1)
            log_var = tf.concat([inst_log_var, class_log_var], axis=-1)
            latents = sampling(mu=mean, logVar=log_var)

            loss_sincos_kl, loss_sincos_mse, loss_sincos_1 = self._poseLoss(
                sin_gt=sin_gt, cos_gt=cos_gt, rad_var_gt=self._rad_var,
                sin=sin_mean, cos=cos_mean, rad_log_var=rad_log_var)

            inst_mean_prior, inst_log_var_prior = self._priornet_inst(tf.concat([class_list, inst_list], axis=-1), training=True)
            class_mean_prior, class_log_var_prior = self._priornet_cl(class_list, training=True)
            mean_prior = tf.concat([inst_mean_prior, class_mean_prior], axis=-1)
            log_var_prior = tf.concat([inst_log_var_prior, class_log_var_prior], axis=-1)
            output_images = self._decoder(latents, training=True)

            loss_shape = binary_loss(xPred=output_images, xTarget=output_images_gt, gamma=0.60)
            loss_latent_kl = kl_loss(mean=mean, logVar=log_var, mean_target=mean_prior, logVar_target=log_var_prior)
            loss_inst_prior_reg = regulizer_loss(z_mean=inst_mean_prior, z_logVar=inst_log_var_prior,
                                                  dist_in_z_space=5.0 * 8, class_input=class_list)
            loss_class_prior_reg = regulizer_loss(z_mean=class_mean_prior, z_logVar=class_log_var_prior,
                                                  dist_in_z_space=5.0 * 8)

            loss_sincos_kl = tf.nn.compute_average_loss(loss_sincos_kl, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_sincos_mse = tf.nn.compute_average_loss(loss_sincos_mse, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_sincos_1 = tf.nn.compute_average_loss(loss_sincos_1, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_shape = tf.nn.compute_average_loss(loss_shape, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_latent_kl = tf.nn.compute_average_loss(loss_latent_kl, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_prior_reg = tf.nn.compute_average_loss(loss_inst_prior_reg+loss_class_prior_reg, global_batch_size=self._GLOBAL_BATCH_SIZE)

            total_loss = (
                loss_sincos_kl + 100.0 * loss_sincos_mse + 1000.0 * loss_sincos_1
                + loss_shape
                + loss_latent_kl
                + 0.01 * loss_prior_reg
            )
        trainable_variables = self._encoder_backbone.trainable_variables + self._encoder_head.trainable_variables \
                              + self._decoder.trainable_variables + self._priornet_inst.trainable_variables + self._priornet_cl.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images_gt, xPred=output_images)
        pr = tf.nn.compute_average_loss(TP / (TP + FP + 1e-10), global_batch_size=self._GLOBAL_BATCH_SIZE)
        rc = tf.nn.compute_average_loss(TP / (TP + FN + 1e-10), global_batch_size=self._GLOBAL_BATCH_SIZE)

        return loss_sincos_kl, loss_sincos_mse, loss_sincos_1,\
        loss_shape, loss_latent_kl, loss_prior_reg,\
        pr, rc

    def distributed_fit(self, inputs):
        sckl, scmse, sc1, s, lkl, reg, pr, rc = self._strategy.run(self.fit, args=(inputs,))
        sckl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, sckl, axis=None)
        scmse = self._strategy.reduce(tf.distribute.ReduceOp.SUM, scmse, axis=None)
        sc1 = self._strategy.reduce(tf.distribute.ReduceOp.SUM, sc1, axis=None)
        s = self._strategy.reduce(tf.distribute.ReduceOp.SUM, s, axis=None)
        lkl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, lkl, axis=None)
        reg = self._strategy.reduce(tf.distribute.ReduceOp.SUM, reg, axis=None)

        pr = self._strategy.reduce(tf.distribute.ReduceOp.SUM, pr, axis=None)
        rc = self._strategy.reduce(tf.distribute.ReduceOp.SUM, rc, axis=None)
        return sckl, scmse, sc1, s, lkl, reg, pr, rc

    def saveEncoderBackbone(self, save_path):
        file_name = 'nolbo_encoder_backbone'
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))
    def saveEncoderHead(self, save_path):
        file_name = 'nolbo_encoder_head'
        self._encoder_head.save_weights(os.path.join(save_path, file_name))
    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)
    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))
    def savePriornet(self, save_path):
        file_name_inst = self._prior_inst_str['name']
        file_name_class = self._prior_cl_str['name']
        self._priornet_inst.save_weights(os.path.join(save_path, file_name_inst))
        self._priornet_cl.save_weights(os.path.join(save_path, file_name_class))
    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriornet(save_path=save_path)

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = 'nolbo_encoder_backbone'
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))
    def loadEncoderHead(self, load_path, file_name=None):
        if file_name == None:
            file_name = 'nolbo_encoder_head'
        self._encoder_head.load_weights(os.path.join(load_path, file_name))
    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)
    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))
    def loadPriornet(self, load_path, file_name=None):
        file_name_inst = self._prior_inst_str['name']
        file_name_class = self._prior_cl_str['name']
        self._priornet_inst.load_weights(os.path.join(load_path, file_name_inst))
        self._priornet_cl.load_weights(os.path.join(load_path, file_name_class))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriornet(load_path=load_path)

    def _getEV(self, sin, cos, radLogVar):
        Esin = tf.exp(-tf.exp(radLogVar) / 2.0) * sin
        Ecos = tf.exp(-tf.exp(radLogVar) / 2.0) * cos
        Varsin = 0.5 - 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (1.0 - 2.0 * sin * sin) - tf.exp(
            -tf.exp(radLogVar)) * sin * sin
        Varcos = 0.5 + 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (2.0 * cos * cos - 1.0) - tf.exp(
            -tf.exp(radLogVar)) * cos * cos
        logVarsin = tf.math.log(Varsin + 1e-7)
        logVarcos = tf.math.log(Varcos + 1e-7)
        return Esin, Ecos, logVarsin, logVarcos

    def _poseLoss(self, sin_gt, cos_gt, rad_var_gt, sin, cos, rad_log_var):
        Esin_gt, Ecos_gt, log_var_sin_gt, log_var_cos_gt = self._getEV(
            sin=sin_gt, cos=cos_gt, radLogVar=tf.math.log(rad_var_gt+1e-9))
        Esin_pr, Ecos_pr, log_var_sin_pr, log_var_cos_pr = self._getEV(
            sin=sin, cos=cos, radLogVar=rad_log_var)

        loss_sin_kl = kl_loss(mean=Esin_pr, logVar=log_var_sin_pr, mean_target=Esin_gt, logVar_target=log_var_sin_gt)
        loss_cos_kl = kl_loss(mean=Ecos_pr, logVar=log_var_cos_pr, mean_target=Ecos_gt, logVar_target=log_var_cos_gt)

        sinz = sampling(mu=Esin_pr, logVar=log_var_sin_pr)
        cosz = sampling(mu=Ecos_pr, logVar=log_var_cos_pr)
        loss_sincos_mse = tf.square(sin_gt - sin)/tf.exp(log_var_sin_gt) \
                                + tf.square(cos_gt - cos)/tf.exp(log_var_cos_gt) \
                                + tf.square(rad_log_var - tf.math.log(rad_var_gt+1e-9)) \
                                + tf.square(sin_gt - sinz) + tf.square(cos_gt - cosz)
                                # + tf.square(Esin_gt - Esin_pr) + tf.square(Ecos_gt - Ecos_pr) \
                                # + tf.square(self._ori_sin_gt_tile - self._ori_sin_mean_tile)+ tf.square(self._ori_cos_gt_tile - self._ori_cos_mean_tile) \
        # self._loss_sincos_mse = tf.square(self._ori_sin_gt_tile - self._ori_sin_mean_tile) \
        #                         + tf.square(self._ori_cos_gt_tile - self._ori_cos_mean_tile) \
        #                         + tf.square(self._rad_log_var_tile - tf.math.log(self._rad_var+1e-9))
        loss_sincos_1 = tf.square(tf.square(sin)+tf.square(cos) - 1.0)

        return loss_sin_kl + loss_cos_kl, loss_sincos_mse, loss_sincos_1


class pretrain_integrated(object):
    def __init__(self,
                 backbone_style=None,
                 encoder_backbone=None,
                 decoder_structure=None,
                 prior_class_structure=None,
                 prior_inst_structure=None,
                 BATCH_SIZE_PER_REPLICA_nolbo=32,
                 BATCH_SIZE_PER_REPLICA_classifier=64,
                 strategy=None,
                 learning_rate = 1e-4
                 ):
        self._encoder_backbone = encoder_backbone
        self._backbone_style = backbone_style
        self._rad_var = (15.0/180.0 * 3.141593) ** 2
        self._dec_str = decoder_structure
        self._prior_cl_str = prior_class_structure
        self._prior_inst_str = prior_inst_structure

        self._strategy = strategy
        # self._strategy = tf.distribute.MirroredStrategy()
        self._GLOBAL_BATCH_SIZE_nolbo = BATCH_SIZE_PER_REPLICA_nolbo * self._strategy.num_replicas_in_sync
        self._GLOBAL_BATCH_SIZE_classifier = BATCH_SIZE_PER_REPLICA_classifier * self._strategy.num_replicas_in_sync

        with self._strategy.scope():
            self._buildModel()
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            # self._optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
            # self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def _buildModel(self):
        print('build models....')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name='backbone', activation='elu')

        # ================================= set model head
        self._encoder_head_imagenet = darknet.head2D(name='head_imagenet',
                                                  input_shape=self._encoder_backbone.output_shape[1:],
                                                  output_dim=1000,
                                                  filter_num_list=[],
                                                  filter_size_list=[],
                                                  last_pooling='max', activation='elu')
        self._encoder_head_place365 = darknet.head2D(name='head_imagenet',
                                                  input_shape=self._encoder_backbone.output_shape[1:],
                                                  output_dim=365,
                                                  filter_num_list=[],
                                                  filter_size_list=[],
                                                  last_pooling='max', activation='elu')

        # ==============set encoder head
        self._encoder_head_nolbo = darknet.head2D(name='head_nolbo',
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=(2*3+3 + 2*(8+8)),
                                            filter_num_list=[1024, 1024, 1024],
                                            filter_size_list=[3, 3, 3],
                                            last_pooling='max', activation='elu')
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        self._priornet_cl = priornet.priornet(structure=self._prior_cl_str)
        self._priornet_inst = priornet.priornet(structure=self._prior_inst_str)
        print('done')

    # @tf.function
    def _lossObject(self, y_target, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        loss = -tf.reduce_sum(y_target * tf.math.log(y_pred + 1e-9), axis=-1)
        return tf.nn.compute_average_loss(loss, global_batch_size=self._GLOBAL_BATCH_SIZE_classifier)

    # @tf.function
    def _evaluation(self, y_target, y_pred):
        gt = tf.argmax(y_target, axis=-1)
        pr = tf.argmax(y_pred, axis=-1)
        equality = tf.equal(pr, gt)
        acc_top1 = tf.cast(equality, tf.float32)
        acc_top5 = tf.cast(
            tf.math.in_top_k(
                predictions=y_pred,
                targets=gt, k=5
            ),
            tf.float32)
        return tf.nn.compute_average_loss(
            acc_top1, global_batch_size=self._GLOBAL_BATCH_SIZE_classifier
        ), tf.nn.compute_average_loss(
            acc_top5, global_batch_size=self._GLOBAL_BATCH_SIZE_classifier
        )

    def fit(self, inputs_imagenet, inputs_place365, inputs_nolbo):
        input_images_imagenet, class_list_imagenet = inputs_imagenet
        input_images_place365, class_list_place365 = inputs_place365
        class_list, inst_list, sin_gt, cos_gt, input_images, output_images_gt = inputs_nolbo
        with tf.GradientTape() as tape:
            class_list_imagenet_pred = self._encoder_head_imagenet(self._encoder_backbone(input_images_imagenet, training=True), training=True)
            pred_loss_imagenet = self._lossObject(y_target=class_list_imagenet, y_pred=class_list_imagenet_pred)
            class_list_place365_pred = self._encoder_head_place365(self._encoder_backbone(input_images_place365, training=True), training=True)
            pred_loss_place365 = self._lossObject(y_target=class_list_place365, y_pred=class_list_place365_pred)

            # get encoder output
            enc_output = self._encoder_head_nolbo(self._encoder_backbone(input_images, training=True), training=True)
            inst_mean = enc_output[..., :8]
            inst_log_var = enc_output[..., 8:16]
            class_mean = enc_output[..., 16:16+8]
            class_log_var = enc_output[..., 16+8:16+16]
            sin_mean = tf.tanh(enc_output[..., 16+16: 16+16+3])
            cos_mean = tf.tanh(enc_output[..., 16+16+3:16+16+3+3])
            rad_log_var = enc_output[..., 16+16+3+3:]
            mean = tf.concat([inst_mean, class_mean], axis=-1)
            log_var = tf.concat([inst_log_var, class_log_var], axis=-1)
            latents = sampling(mu=mean, logVar=log_var)

            loss_sincos_kl, loss_sincos_mse, loss_sincos_1 = self._poseLoss(
                sin_gt=sin_gt, cos_gt=cos_gt, rad_var_gt=self._rad_var,
                sin=sin_mean, cos=cos_mean, rad_log_var=rad_log_var)

            inst_mean_prior, inst_log_var_prior = self._priornet_inst(tf.concat([class_list, inst_list], axis=-1), training=True)
            class_mean_prior, class_log_var_prior = self._priornet_cl(class_list, training=True)
            mean_prior = tf.concat([inst_mean_prior, class_mean_prior], axis=-1)
            log_var_prior = tf.concat([inst_log_var_prior, class_log_var_prior], axis=-1)
            output_images = self._decoder(latents, training=True)

            loss_shape = binary_loss(xPred=output_images, xTarget=output_images_gt, gamma=0.60)
            loss_latent_kl = kl_loss(mean=mean, logVar=log_var, mean_target=mean_prior, logVar_target=log_var_prior)
            loss_inst_prior_reg = regulizer_loss(z_mean=inst_mean_prior, z_logVar=inst_log_var_prior,
                                                  dist_in_z_space=5.0 * 8, class_input=class_list)
            loss_class_prior_reg = regulizer_loss(z_mean=class_mean_prior, z_logVar=class_log_var_prior,
                                                  dist_in_z_space=5.0 * 8)

            loss_sincos_kl = tf.nn.compute_average_loss(loss_sincos_kl, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_sincos_mse = tf.nn.compute_average_loss(loss_sincos_mse, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_sincos_1 = tf.nn.compute_average_loss(loss_sincos_1, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_shape = tf.nn.compute_average_loss(loss_shape, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_latent_kl = tf.nn.compute_average_loss(loss_latent_kl, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_prior_reg = tf.nn.compute_average_loss(loss_inst_prior_reg+loss_class_prior_reg, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)

            total_loss = (
                pred_loss_imagenet
                + pred_loss_place365
                + loss_sincos_kl + 100.0 * loss_sincos_mse + 1000.0 * loss_sincos_1
                + loss_shape
                + loss_latent_kl
                + 0.01 * loss_prior_reg
            )
        trainable_variables = self._encoder_backbone.trainable_variables\
                              + self._encoder_head_imagenet.trainable_variables + self._encoder_head_place365.trainable_variables + self._encoder_head_nolbo.trainable_variables \
                              + self._decoder.trainable_variables + self._priornet_inst.trainable_variables + self._priornet_cl.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        acc_top1_imagenet, acc_top5_imagenet = self._evaluation(y_target=class_list_imagenet, y_pred=class_list_imagenet_pred)
        acc_top1_place365, acc_top5_place365 = self._evaluation(y_target=class_list_place365, y_pred=class_list_place365_pred)

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images_gt, xPred=output_images)
        pr = tf.nn.compute_average_loss(TP / (TP + FP + 1e-10), global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
        rc = tf.nn.compute_average_loss(TP / (TP + FN + 1e-10), global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)

        return pred_loss_imagenet, pred_loss_place365, acc_top1_imagenet, acc_top1_place365, \
        acc_top5_imagenet, acc_top5_place365, loss_sincos_mse, loss_shape, pr, rc

    def distributed_fit(self, inputs_imagenet, inputs_place365, inputs_nolbo):
        limage, lplace, t1image, t1place, t5image, t5place, lscmse, lshape, pr, rc = self._strategy.run(self.fit, args=(inputs_imagenet, inputs_place365, inputs_nolbo,))
        limage = self._strategy.reduce(tf.distribute.ReduceOp.SUM, limage, axis=None)
        lplace = self._strategy.reduce(tf.distribute.ReduceOp.SUM, lplace, axis=None)
        t1image = self._strategy.reduce(tf.distribute.ReduceOp.SUM, t1image, axis=None)
        t1place = self._strategy.reduce(tf.distribute.ReduceOp.SUM, t1place, axis=None)
        t5image = self._strategy.reduce(tf.distribute.ReduceOp.SUM, t5image, axis=None)
        t5place = self._strategy.reduce(tf.distribute.ReduceOp.SUM, t5place, axis=None)
        lscmse = self._strategy.reduce(tf.distribute.ReduceOp.SUM, lscmse, axis=None)
        lshape = self._strategy.reduce(tf.distribute.ReduceOp.SUM, lshape, axis=None)

        pr = self._strategy.reduce(tf.distribute.ReduceOp.SUM, pr, axis=None)
        rc = self._strategy.reduce(tf.distribute.ReduceOp.SUM, rc, axis=None)
        return limage, lplace, t1image, t1place, t5image, t5place, lscmse, lshape, pr, rc

    def saveEncoderBackbone(self, save_path):
        file_name = 'backbone'
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))
    def saveEncoderHead(self, save_path):
        self._encoder_head_imagenet.save_weights(os.path.join(save_path, 'head_imagenet'))
        self._encoder_head_place365.save_weights(os.path.join(save_path, 'head_place365'))
        self._encoder_head_nolbo.save_weights(os.path.join(save_path, 'head_nolbo'))
    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)
    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))
    def savePriornet(self, save_path):
        file_name_inst = self._prior_inst_str['name']
        file_name_class = self._prior_cl_str['name']
        self._priornet_inst.save_weights(os.path.join(save_path, file_name_inst))
        self._priornet_cl.save_weights(os.path.join(save_path, file_name_class))
    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriornet(save_path=save_path)

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = 'backbone'
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))
    def loadEncoderHead(self, load_path):
        self._encoder_head_imagenet.load_weights(os.path.join(load_path, 'head_imagenet'))
        self._encoder_head_place365.load_weights(os.path.join(load_path, 'head_place365'))
        self._encoder_head_nolbo.load_weights(os.path.join(load_path, 'head_nolbo'))
    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)
    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))
    def loadPriornet(self, load_path, file_name=None):
        file_name_inst = self._prior_inst_str['name']
        file_name_class = self._prior_cl_str['name']
        self._priornet_inst.load_weights(os.path.join(load_path, file_name_inst))
        self._priornet_cl.load_weights(os.path.join(load_path, file_name_class))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriornet(load_path=load_path)

    def _getEV(self, sin, cos, radLogVar):
        Esin = tf.exp(-tf.exp(radLogVar) / 2.0) * sin
        Ecos = tf.exp(-tf.exp(radLogVar) / 2.0) * cos
        Varsin = 0.5 - 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (1.0 - 2.0 * sin * sin) - tf.exp(
            -tf.exp(radLogVar)) * sin * sin
        Varcos = 0.5 + 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (2.0 * cos * cos - 1.0) - tf.exp(
            -tf.exp(radLogVar)) * cos * cos
        logVarsin = tf.math.log(Varsin + 1e-7)
        logVarcos = tf.math.log(Varcos + 1e-7)
        return Esin, Ecos, logVarsin, logVarcos

    def _poseLoss(self, sin_gt, cos_gt, rad_var_gt, sin, cos, rad_log_var):
        Esin_gt, Ecos_gt, log_var_sin_gt, log_var_cos_gt = self._getEV(
            sin=sin_gt, cos=cos_gt, radLogVar=tf.math.log(rad_var_gt+1e-9))
        Esin_pr, Ecos_pr, log_var_sin_pr, log_var_cos_pr = self._getEV(
            sin=sin, cos=cos, radLogVar=rad_log_var)

        loss_sin_kl = kl_loss(mean=Esin_pr, logVar=log_var_sin_pr, mean_target=Esin_gt, logVar_target=log_var_sin_gt)
        loss_cos_kl = kl_loss(mean=Ecos_pr, logVar=log_var_cos_pr, mean_target=Ecos_gt, logVar_target=log_var_cos_gt)

        sinz = sampling(mu=Esin_pr, logVar=log_var_sin_pr)
        cosz = sampling(mu=Ecos_pr, logVar=log_var_cos_pr)
        loss_sincos_mse = tf.square(sin_gt - sin)/tf.exp(log_var_sin_gt) \
                                + tf.square(cos_gt - cos)/tf.exp(log_var_cos_gt) \
                                + tf.square(rad_log_var - tf.math.log(rad_var_gt+1e-9)) \
                                + tf.square(sin_gt - sinz) + tf.square(cos_gt - cosz)
                                # + tf.square(Esin_gt - Esin_pr) + tf.square(Ecos_gt - Ecos_pr) \
                                # + tf.square(self._ori_sin_gt_tile - self._ori_sin_mean_tile)+ tf.square(self._ori_cos_gt_tile - self._ori_cos_mean_tile) \
        # self._loss_sincos_mse = tf.square(self._ori_sin_gt_tile - self._ori_sin_mean_tile) \
        #                         + tf.square(self._ori_cos_gt_tile - self._ori_cos_mean_tile) \
        #                         + tf.square(self._rad_log_var_tile - tf.math.log(self._rad_var+1e-9))
        loss_sincos_1 = tf.square(tf.square(sin)+tf.square(cos) - 1.0)

        return loss_sin_kl + loss_cos_kl, loss_sincos_mse, loss_sincos_1



















