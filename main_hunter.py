import numpy as np
import tensorflow as tf
import pdb
from trainer import *
from trainer256 import *
from config import get_config
from utils import prepare_dirs_and_logger, save_config
import cv2
import pdb, os

def main(config):
    prepare_dirs_and_logger(config)

    if config.gpu>-1:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu)

    config.data_format = 'NHWC'

    if 1==config.model: 
        trainer = PG2(config)
        trainer.init_net()
    elif 11==config.model:
        trainer = PG2_256(config)
        trainer.init_net()
        
    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        # if not config.load_path:
        #     raise Exception("[!] You should specify `load_path` to load a pretrained model")
        input_path = './hunter_test/df002.png'
        pose_path = './hunter_test/ultraman1.npy'
        x = cv2.imread(input_path)
        p = np.load(pose_path)
        #pp = p[0:18,:,:]
        trainer.generate_hunter(x, p)

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
