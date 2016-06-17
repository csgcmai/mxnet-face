import mxnet as mx
import argparse
import logging
import scipy.io as sio
import numpy as np
import h5py
from lightened_cnn import lightened_cnn_b_feature

# don't use -n and -s, which are resevered for the distributed training
parser = argparse.ArgumentParser(description='train an image classifer on imagenet')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--gpus', type=str, default='1',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--outputs', type=str, default='linear',
                    help='name of the outputs, e.g "linear"')
args = parser.parse_args()

# network
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

batch_size = args.batch_size
# train
devs = mx.cpu() if args.gpus is None else [
    mx.gpu(int(i)) for i in args.gpus.split(',')]

net, tmp_arg_params,tmp_aux_params = mx.model.load_checkpoint(args.model_prefix, args.load_epoch)

model_trained = mx.model.FeedForward(net, ctx=devs,arg_params=tmp_arg_params,
    aux_params=tmp_aux_params, begin_epoch=args.load_epoch, numpy_batch_size=batch_size)

model_internal = model_trained.symbol.get_internals()
model_names = {out+'_output' for out in args.outputs.split(',')}#('sign0', 'mae','fullyconnected1')


xDataPrefix = '/dev/shm/maiguang/rep_rgbimg_'
xDataKey = 'data'
saveprefix = '/user/maiguang/project/database/rep_reludata_'
data_src = {'casia_webface_openface',}

saveprefix = '/user/maiguang/project/database/drop1data_'
xDataPrefix = '/user/maiguang/project/database/img_'
data_src = {'lfw', 'frgc'}

model_internal = model_trained.symbol.get_internals()
print(model_internal.list_outputs())
data_label_flag = False
for data_name in data_src:
    f = h5py.File(saveprefix+data_name+'.mat','w')

    validX = np.array(h5py.File(xDataPrefix+data_name+'.mat','r')[xDataKey])
    #labelX = np.array(h5py.File(xDataPrefix+data_name+'.mat','r')['label'])

    f.create_dataset('data',data=validX)
    #f.create_dataset('label',data=labelX)
        
    for net_internal in model_names:
        print(net_internal)
        model_itn = mx.model.FeedForward(ctx=devs,
            numpy_batch_size=batch_size, 
            symbol=model_internal[net_internal], 
            arg_params=model_trained.arg_params, 
            aux_params=model_trained.aux_params, 
            allow_extra_params=True)
        
        feature_itn = model_itn.predict(validX/255.0)
        feature_itn = np.float32(feature_itn)
        del model_itn
        f.create_dataset(net_internal.split('_')[0],data=feature_itn)
    f.close()

