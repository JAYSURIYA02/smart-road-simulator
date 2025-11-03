#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image and save lane overlay image
"""
import argparse
import os.path as ops
import time
import cv2
import numpy as np
import tensorflow as tf

# Ensure TF1.x style for LaneNet
tf.compat.v1.disable_eager_execution()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger


CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


def args_str2bool(arg_value):
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The input image path')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--with_lane_fit', type=args_str2bool, default=True, help='If need to do lane fit')
    parser.add_argument('--save_path', type=str, default='lane_result.jpg', help='Path to save the lane mask image')
    return parser.parse_args()


def test_lanenet(image_path, weights_path, save_path= "lane_result1.jpg", with_lane_fit=True):
    """Run LaneNet and save lane overlay image"""
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    LOG.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image.copy()
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    # TF1.x placeholder
    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # TensorFlow session setup
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)

    with tf.compat.v1.variable_scope('moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.compat.v1.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=with_lane_fit,
            data_source='tusimple'
        )

        mask_image = postprocess_result['mask_image']
        cv2.imwrite(save_path, mask_image)
        LOG.info(f"✅ Lane mask image saved at: {save_path}")

        if with_lane_fit and 'fit_params' in postprocess_result:
            lane_params = postprocess_result['fit_params']
            LOG.info('Model fitted {:d} lanes'.format(len(lane_params)))
            for i, params in enumerate(lane_params):
                LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, params))

    sess.close()


if __name__ == '__main__':
    args = init_args()
    test_lanenet(
        image_path=args.image_path,
        weights_path=args.weights_path,
        save_path=args.save_path,
        with_lane_fit=args.with_lane_fit
    )
