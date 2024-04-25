#!bin/bash

python3 coco_converter.py --coco_json train/_annotations.coco.json --image_dir ./train/ --output_dir $PWD

python3 data_splitter.py ./training/train.txt $0 $1 $2
