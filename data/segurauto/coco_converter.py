import argparse
import os
import os.path as ops
import numpy as np
import cv2
from pycocotools.coco import COCO


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_json', type=str, help='Path to your COCO dataset JSON file')
    parser.add_argument('--image_dir', type=str, help='Path to the directory containing images')
    parser.add_argument('--output_dir', type=str, help='Output directory for generated images')
    return parser.parse_args()


def create_output_directories(output_dir):
    binary_dir = ops.join(output_dir, 'gt_binary_image')
    instance_dir = ops.join(output_dir, 'gt_instance_image')
    os.makedirs(binary_dir, exist_ok=True)
    os.makedirs(instance_dir, exist_ok=True)
    return binary_dir, instance_dir


def process_coco_dataset(coco_json, image_dir, binary_dir, instance_dir):
    coco = COCO(coco_json)

    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
        image_path = ops.join(image_dir, image_info['file_name'])
        image = cv2.imread(image_path)

        anns_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(anns_ids)

        binary_mask = np.zeros(image.shape[:2], np.uint8)
        instance_mask = np.zeros(image.shape[:2], np.uint8)

        for idx, annotation in enumerate(annotations):
            if type(annotation['segmentation']) == list: 
                for seg in annotation['segmentation']:
                    pts = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(binary_mask, [pts], 255)
                    cv2.fillPoly(instance_mask, [pts], idx * 100 + 20)  
                    
        
        base_name = ops.splitext(image_info['file_name'])[0]
        binary_path = ops.join(binary_dir, f'{base_name}.jpg')
        instance_path = ops.join(instance_dir, f'{base_name}.jpg')
        cv2.imwrite(binary_path, binary_mask)
        cv2.imwrite(instance_path, instance_mask)



def gen_train_sample(src_dir, b_gt_image_dir, i_gt_image_dir, image_dir):
    with open('{:s}/training/train.txt'.format(src_dir), 'w') as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.jpg'):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                print('{:s}'.format(image_name))
                continue
            else:
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path)
                file.write(info + '\n')
    return

if __name__ == '__main__':
    args = init_args()

    binary_dir, instance_dir = create_output_directories(args.output_dir)
    process_coco_dataset(args.coco_json, args.image_dir, binary_dir, instance_dir)
    gen_train_sample(args.output_dir, binary_dir, instance_dir, args.image_dir)
