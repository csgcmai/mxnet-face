#!/usr/bin/env sh

raw_data_path=/user/maiguang/Downloads/CASIA-WebFace/CASIA-WebFace
align_data_path=/research/prip-maiguang/Database/CASIA-Webface-r108
makelist_path=/user/maiguang/mxnet/tools/make_list.py
# the number threads used for align data, you shold change this depend on your environment
num_process=8
landmarks=innerEyesAndBottomLip
face_size=108
ts=0.1
list_name=casia_r108
rec_name=casia_r108
im2rec_path=/user/maiguang/mxnet/bin/im2rec

# step1:align the face iamge
if ! [ -e $align_data_path ];then
   mkdir -p $align_data_path
   for N in $(seq $num_process);do
	  echo "the sub-process is : $N"
	  python ../util/align_face.py $raw_data_path align $landmarks $align_data_path --ts $ts --size $face_size &
   done
else
   echo "$align_data_path already exist."
fi
wait
echo "Align face image done"

# step2: generate .lst for im2rec
if ! [ -e ${list_name}_train.lst ];then
   python -u $makelist_path $align_data_path $list_name --exts .png --train_ratio 0.95 --recursive True
else
   echo ".lst file for training already exist."
fi
echo "generated .lst file done"

# step3: use img2rec to generate .rec file for training
if ! [ -e ${rec_name}_train.rec ]; then
	$im2rec_path ${list_name}_train.lst $align_data_path/ ${rec_name}_train.rec encoding='.png' &
	$im2rec_path ${list_name}_val.lst $align_data_path/ ${rec_name}_val.rec encoding='.png' &
else
	echo "$rec_name already exist."
fi
wait
echo "generate .rec done"

# step4: trainig the model for face recognition
#python -u lightened_cnn.py --gpus 2,3,4,5,6,7
python -u lightened_cnn.py --gpus 0,1
echo "trining done!"
