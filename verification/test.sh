#!/usr/bin/env sh

align_data_path=/user/maiguang/Downloads/LFW/lfw_deepfunneled_aligned_openface
model_prefix=../model/lightened_cnn/lightened_cnn
epoch=60
# evaluate on lfw
python lfw.py --lfw-align $align_data_path --model-prefix $model_prefix --epoch $epoch

# epoch = 58, mean is:0.9410, var is:0.0110
# epoch = 66, mean is:0.9463, var is:0.0098
# epoch = 74, mean is:0.9467, var is:0.0114
# epoch = 82, mean is:0.9443, var is:0.0097
# epoch = 96, mean is:0.9465, var is:0.0090

# newly cropped
# epoch = 51, mean is:0.9455, var is:0.0083
# epoch = 58, mean is:0.9443, var is:0.0110

