import os.path as ops
import argparse
import time

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils

tf.compat.v1.disable_eager_execution()

CFG = parse_config_utils.lanenet_cfg

def get_lanenet_output(image, weights_path):
    """
    Performs inference on a single image and returns the raw network outputs.

    Args:
        image_path: Path to the input image.
        weights_path: Path to the pre-trained model weights file.

    Returns:
        A tuple containing:
            - binary_seg_image: The binary segmentation output.
            - instance_seg_image: The instance segmentation output.
    """

    # assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    # Set sess configuration
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.compat.v1.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.compat.v1.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.compat.v1.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )

    return binary_seg_image, instance_seg_image 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LaneNet inference on camera')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the model weights file')
    args = parser.parse_args()

    vid = cv2.VideoCapture(4)

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    saver = tf.compat.v1.train.Saver()

    # Set sess configuration
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.compat.v1.Session(config=sess_config)

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    while True:
        ret, frame = vid.read() 

        try:
            with sess.as_default():

                saver.restore(sess=sess, save_path=args.weights_path)

                image_vis = frame
                image = cv2.resize(frame, (512, 256), interpolation=cv2.INTER_LINEAR)
                image = image / 127.5 - 1.0

                start_time = time.time()
                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )
                
                postprocess_result = postprocessor.postprocess(
                    binary_seg_result=binary_seg_image[0],
                    instance_seg_result=instance_seg_image[0],
                    with_lane_fit=False,
                    source_image=image_vis
                )
                end_time = time.time()

                mask_image = postprocess_result['mask_image']

                inference_time = end_time - start_time
                fps = 1.0 / inference_time

                binary_mask = (binary_seg_image[0] * 255).astype(np.uint8)  
                instance_mask = (instance_seg_image[0][:, :, (2, 1, 0)]).astype(np.uint8)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(mask_image, 
                            'FPS: {:.1f}'.format(fps),  
                            (0, 0),  
                            font, 1, (0, 255, 0), 2, cv2.LINE_AA) 

                cv2.imshow('frame', )
                # print(fps)
                

        except:
            cv2.imshow('original frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  

    vid.release() 
    cv2.destroyAllWindows() 

    

