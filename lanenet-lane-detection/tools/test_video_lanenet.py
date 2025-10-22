#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
LaneNet Video Inference (CPU-friendly, live display with proper aspect ratio)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import argparse
import cv2
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_video_test')


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
    return output_arr.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--weights_path', type=str, required=True, help='LaneNet weights path (.ckpt)')
    parser.add_argument('--with_lane_fit', type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help='Whether to fit lanes')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video: {}".format(args.video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # LaneNet setup
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # Restore trained weights
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess=sess, save_path=args.weights_path)
    LOG.info("Weights restored successfully.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    LOG.info("Total frames to process: {}".format(frame_count))

    for idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        image_vis = frame
        h_orig, w_orig = frame.shape[:2]

        # Resize to fixed 256x512 for the model
        image = cv2.resize(frame, (512, 256), interpolation=cv2.INTER_LINEAR)
        image_input = image / 127.5 - 1.0  # normalize

        # Run inference
        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image_input]}
        )

        # Postprocess
        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=args.with_lane_fit,
            data_source='tusimple'
        )

        mask_image = postprocess_result['mask_image']

        # Resize mask back to original frame size for display
        mask_resized = cv2.resize(mask_image, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Convert to BGR if needed
        if mask_resized.ndim == 3 and mask_resized.shape[2] == 3:
            mask_bgr = cv2.cvtColor(np.uint8(mask_resized), cv2.COLOR_RGB2BGR)
        else:
            mask_bgr = cv2.cvtColor(np.uint8(mask_resized), cv2.COLOR_GRAY2BGR)

        # Overlay
        display_frame = cv2.addWeighted(image_vis, 0.7, mask_bgr, 0.3, 0)

        cv2.imshow('LaneNet Output', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if idx % 10 == 0:
            LOG.info("Processed frame {}/{}".format(idx, frame_count))


if __name__ == '__main__':
    main()
