# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
import os
from copy import deepcopy
import shutil
import numpy as np

from .utils import load_data,Logger
from .model import Model
import os.path as osp
import sys
def initialize_data(config, train=False, test=False):
    print("Initializing data source...")
    train_source, test_source = load_data(**config['data'], cache=(train or test))
    if train:
        print("Loading training data...")
        train_source.load_all_data()
    if test:
        print("Loading test data...")
        test_source.load_all_data()
    print("Data initialization complete.")
    return train_source, test_source


def initialize_model(config, train_source, test_source,is_test):
    print("Initializing model...")
    data_config = config['data']
    model_config = config['model']
    model_param = deepcopy(model_config)
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_config['pid_num']
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))
    if is_test:
        sys.stdout = Logger(osp.join(model_param['save_name'], '_log_test.txt'))
    else:
        sys.stdout = Logger(osp.join(model_param['save_name'], '_log_train.txt'))
    m = Model(**model_param)
    print("Model initialization complete.")
    return m, model_param['save_name']


def initialization(config, train=False, test=False,is_test=False):
    print("Initialzing...")
    WORK_PATH = config['WORK_PATH']
    if not is_test:
        shutil.copy('./model/model.py',WORK_PATH+'/model.py')
        shutil.copy('./model/network/gaittrans.py',WORK_PATH+'/gaittrans.py')
        shutil.copy('./model/network/basic_blocks.py',WORK_PATH+'/basic_blocks.py')
        shutil.copy('./model/network/vgg_c3d.py',WORK_PATH+'/vgg_c3d.py')
        shutil.copy('./model/network/motion_flow.py',WORK_PATH+'/motion_flow.py')

    if os.getcwd().split('/')[-1]!=WORK_PATH.split('/')[1]:
        os.chdir(WORK_PATH)
    #os.chdir(WORK_PATH)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    train_source, test_source = initialize_data(config, train, test)
    return initialize_model(config, train_source, test_source,is_test)