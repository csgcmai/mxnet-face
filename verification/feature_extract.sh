#!/usr/bin/env sh

align_data_path=/user/maiguang/Downloads/LFW/lfw_deepfunneled_aligned_openface
model_prefix=../model/lightened_cnn_2/lightened_cnn
output=pool1_wxnet,pool2_wxnet,pool3_wxnet,pool4_wxnet,fc1_wxnet,fc2_wxnet
output=drop1_wxnet
epoch=60
# extract features 
python out_data.py  --model-prefix $model_prefix --load-epoch $epoch --outputs $output

