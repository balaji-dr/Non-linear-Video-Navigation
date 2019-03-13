import numpy as np
import sonnet as snt
import tensorflow as tf

from luminoth.models.fasterrcnn.rcnn import RCNN
from luminoth.models.fasterrcnn.rpn import RPN
from luminoth.models.base import TruncatedBaseNetwork
from luminoth.utils.anchors import generate_anchors_reference
from luminoth.utils.vars import VAR_LOG_LEVELS, variable_summaries


class FasterRCNN(snt.AbstractModule):
    def __init__(self, config, name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)

        self._config = config

        self._num_classes = config.model.network.num_classes

        # Generate network with RCNN 
        self._with_rcnn = config.model.network.with_rcnn

        self._debug = config.train.debug
        self._seed = config.train.seed

        self._anchor_base_size = config.model.anchors.base_size
        self._anchor_scales = np.array(config.model.anchors.scales)
        self._anchor_ratios = np.array(config.model.anchors.ratios)
        self._anchor_stride = config.model.anchors.stride

        self._anchor_reference = generate_anchors_reference(
            self._anchor_base_size, self._anchor_ratios, self._anchor_scales
        )

        self._num_anchors = self._anchor_reference.shape[0]

        # Weights used to sum each of the losses of the submodules
        self._rpn_cls_loss_weight = config.model.loss.rpn_cls_loss_weight
        self._rpn_reg_loss_weight = config.model.loss.rpn_reg_loss_weights

        self._rcnn_cls_loss_weight = config.model.loss.rcnn_cls_loss_weight
        self._rcnn_reg_loss_weight = config.model.loss.rcnn_reg_loss_weights
        self._losses_collections = ['fastercnn_losses']

        self.base_network = TruncatedBaseNetwork(config.model.base_network)

    def _build(self, image, gt_boxes=None, is_training=False):

        if gt_boxes is not None:
            gt_boxes = tf.cast(gt_boxes, tf.float32)

        image.set_shape((None, None, 3))

        conv_feature_map = self.base_network(
            tf.expand_dims(image, 0), is_training=is_training
        )

        # The RPN submodule which generates proposals of objects.
        self._rpn = RPN(
            self._num_anchors, self._config.model.rpn,
            debug=self._debug, seed=self._seed
        )
        if self._with_rcnn:
            self._rcnn = RCNN(
                self._num_classes, self._config.model.rcnn,
                debug=self._debug, seed=self._seed
            )

        image_shape = tf.shape(image)[0:2]

        variable_summaries(
            conv_feature_map, 'conv_feature_map', 'reduced'
        )

        all_anchors = self._generate_anchors(tf.shape(conv_feature_map))
        rpn_prediction = self._rpn(
            conv_feature_map, image_shape, all_anchors,
            gt_boxes=gt_boxes, is_training=is_training
        )

        prediction_dict = {
            'rpn_prediction': rpn_prediction,
        }

        if self._debug:
            prediction_dict['image'] = image
            prediction_dict['image_shape'] = image_shape
            prediction_dict['all_anchors'] = all_anchors
            prediction_dict['anchor_reference'] = tf.convert_to_tensor(
                self._anchor_reference
            )
            if gt_boxes is not None:
                prediction_dict['gt_boxes'] = gt_boxes
            prediction_dict['conv_feature_map'] = conv_feature_map

        if self._with_rcnn:
            proposals = tf.stop_gradient(rpn_prediction['proposals'])
            classification_pred = self._rcnn(
                conv_feature_map, proposals,
                image_shape, self.base_network,
                gt_boxes=gt_boxes, is_training=is_training
            )

            prediction_dict['classification_prediction'] = classification_pred

        return prediction_dict

    def loss(self, prediction_dict, return_all=False):
        with tf.name_scope('losses'):
            rpn_loss_dict = self._rpn.loss(
                prediction_dict['rpn_prediction']
            )

            rpn_loss_dict['rpn_cls_loss'] = (
                rpn_loss_dict['rpn_cls_loss'] * self._rpn_cls_loss_weight)
            rpn_loss_dict['rpn_reg_loss'] = (
                rpn_loss_dict['rpn_reg_loss'] * self._rpn_reg_loss_weight)

            prediction_dict['rpn_loss_dict'] = rpn_loss_dict

            if self._with_rcnn:
                rcnn_loss_dict = self._rcnn.loss(
                    prediction_dict['classification_prediction']
                )

                rcnn_loss_dict['rcnn_cls_loss'] = (
                    rcnn_loss_dict['rcnn_cls_loss'] *
                    self._rcnn_cls_loss_weight
                )
                rcnn_loss_dict['rcnn_reg_loss'] = (
                    rcnn_loss_dict['rcnn_reg_loss'] *
                    self._rcnn_reg_loss_weight
                )

                prediction_dict['rcnn_loss_dict'] = rcnn_loss_dict
            else:
                rcnn_loss_dict = {}

            all_losses_items = (
                list(rpn_loss_dict.items()) + list(rcnn_loss_dict.items()))

            for loss_name, loss_tensor in all_losses_items:
                tf.summary.scalar(
                    loss_name, loss_tensor,
                    collections=self._losses_collections
                )
                tf.losses.add_loss(loss_tensor)

            regularization_loss = tf.losses.get_regularization_loss()
            no_reg_loss = tf.losses.get_total_loss(
                add_regularization_losses=False
            )
            total_loss = tf.losses.get_total_loss()

            tf.summary.scalar(
                'total_loss', total_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'no_reg_loss', no_reg_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'regularization_loss', regularization_loss,
                collections=self._losses_collections
            )

            if return_all:
                loss_dict = {
                    'total_loss': total_loss,
                    'no_reg_loss': no_reg_loss,
                    'regularization_loss': regularization_loss,
                }

                for loss_name, loss_tensor in all_losses_items:
                    loss_dict[loss_name] = loss_tensor

                return loss_dict

            return total_loss

    def _generate_anchors(self, feature_map_shape):
        with tf.variable_scope('generate_anchors'):
            grid_width = feature_map_shape[2]  # width
            grid_height = feature_map_shape[1]  # height
            shift_x = tf.range(grid_width) * self._anchor_stride
            shift_y = tf.range(grid_height) * self._anchor_stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

            shift_x = tf.reshape(shift_x, [-1])
            shift_y = tf.reshape(shift_y, [-1])

            shifts = tf.stack(
                [shift_x, shift_y, shift_x, shift_y],
                axis=0
            )

            shifts = tf.transpose(shifts)
            all_anchors = (
                np.expand_dims(self._anchor_reference, axis=0) +
                tf.expand_dims(shifts, axis=1)
            )

            all_anchors = tf.reshape(
                all_anchors, (-1, 4)
            )
            return all_anchors

    @property
    def summary(self):
        summaries = [
            tf.summary.merge_all(key='rpn'),
        ]

        summaries.append(
            tf.summary.merge_all(key=self._losses_collections[0])
        )

        if self._with_rcnn:
            summaries.append(tf.summary.merge_all(key='rcnn'))

        return tf.summary.merge(summaries)

    @property
    def vars_summary(self):
        return {
            key: tf.summary.merge_all(key=collection)
            for key, collections in VAR_LOG_LEVELS.items()
            for collection in collections
        }

    def get_trainable_vars(self):
        trainable_vars = snt.get_variables_in_module(self)
        if self._config.model.base_network.trainable:
            pretrained_trainable_vars = self.base_network.get_trainable_vars()
            if len(pretrained_trainable_vars):
                tf.logging.info(
                    'Training {} vars from pretrained module; '
                    'from "{}" to "{}".'.format(
                        len(pretrained_trainable_vars),
                        pretrained_trainable_vars[0].name,
                        pretrained_trainable_vars[-1].name,
                    )
                )
            else:
                tf.logging.info('No vars from pretrained module to train.')
            trainable_vars += pretrained_trainable_vars
        else:
            tf.logging.info('Not training variables from pretrained module')

        return trainable_vars

    def get_base_network_checkpoint_vars(self):
        return self.base_network.get_base_network_checkpoint_vars()

    def get_checkpoint_file(self):
        return self.base_network.get_checkpoint_file()
