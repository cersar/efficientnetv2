import tensorflow as tf
import os
from efficientnetv2.efficientnetv2 import EfficientNetV2
import argparse


def ckpt_convert(model_name,ckpt_path,save_path):
    model = EfficientNetV2(model_name=model_name)
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)

    for var in model.variables:
        v = ckpt_reader.get_tensor(model_name+'/'+var.name.replace(':0',''))
        var.assign(v)

    model.save_weights(os.path.join(save_path, os.path.basename(ckpt_path)+'.h5'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ckpt convert.')
    parser.add_argument('model_name', help='model_name')
    parser.add_argument('ckpt_path', help='ckpt_path')
    parser.add_argument('--save_path', help='save_path',default='./')
    args = parser.parse_args()

    ckpt_convert(args.model_name,args.ckpt_path,args.save_path)
