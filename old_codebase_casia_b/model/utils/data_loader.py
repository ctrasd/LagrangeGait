import os
import os.path as osp

import numpy as np

from .data_set import DataSet
from tqdm import tqdm

def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()
    cnt=0
    for _label in tqdm(sorted(list(os.listdir(dataset_path)))):
        if dataset == 'CASIA-B' and _label == '005':
            continue
        label_path = osp.join(dataset_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):
                _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append(_view)
                    cnt=cnt+1
    
    if dataset == 'CASIA-B' or dataset == 'CASIA-E':
        print(cnt)
        pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
            dataset, pid_num, pid_shuffle))
        # print(pid_fname)
        if not osp.exists(pid_fname):
            pid_list = sorted(list(set(label)))
            if pid_shuffle:
                np.random.shuffle(pid_list)
            pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
            os.makedirs('partition', exist_ok=True)
            np.save(pid_fname, pid_list)
        # pid_list = np.load(pid_fname)
        pid_list = np.load(pid_fname, allow_pickle=True)
        train_list = pid_list[0]
        test_list = pid_list[1]
    else:
        pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
            dataset, pid_num, pid_shuffle))
        # print(pid_fname)
        pid_list = list()
        if not osp.exists(pid_fname):
            for i in range(1,10307):
                if i % 2!=0:
                    # counteven += 1 
                    tem = '%05d'%i
                    pid_list.append(tem)  
            for i in range(1,10307):
                if i % 2==0:
                    # countodd += 1 
                    tem = '%05d'%i
                    pid_list.append(tem)
            pid_list.append('10307')
            # pid_list = sorted(list(set(label)))
            if pid_shuffle:
                np.random.shuffle(pid_list)
            pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
            os.makedirs('partition', exist_ok=True)
            np.save(pid_fname, pid_list)
        # pid_list = np.load(pid_fname)
        pid_list = np.load(pid_fname, allow_pickle=True)
        train_list = pid_list[0]
        test_list = pid_list[1]
        # print(len(train_list))
    print('lentestdata--',len(train_list),len(test_list))
    train_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        [seq_type[i] for i, l in enumerate(label) if l in train_list],
        [view[i] for i, l in enumerate(label)
         if l in train_list],
        cache, resolution, cut=True)
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        cache, resolution, cut=True)
    print('len train,test--',len(train_source),len(test_source))
    # print(train_source[0])
    # print(test_source[0])
    return train_source, test_source

def load_data_lmdb(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()
    cnt=0
    for _label in tqdm(sorted(list(os.listdir(dataset_path)))):
        if dataset == 'CASIA-B' and _label == '005':
            continue
        label_path = osp.join(dataset_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):
                _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_label])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append(_view)
                    cnt=cnt+1
    label_set = set(label)
    seq_type_set = set(seq_type)
    view_set = set(view)
    _ = np.zeros((len(label_set),
                    len(seq_type_set),
                    len(view_set))).astype('int')
    
    _ -= 1
    self.index_dict = xr.DataArray(
        _,
        coords={'label': sorted(label_set),
                'seq_type': sorted(seq_type_set),
                'view': sorted(view_set)},
        dims=['label', 'seq_type', 'view'])    
    for i in range(cnt):
        _label = label[i]
        _seq_type = seq_type[i]
        _view = view[i]
        self.index_dict.loc[_label, _seq_type, _view] = i

    if dataset == 'CASIA-B' or dataset == 'CASIA-E':
        print(cnt)
        pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
            dataset, pid_num, pid_shuffle))
        # print(pid_fname)
        if not osp.exists(pid_fname):
            pid_list = sorted(list(set(label)))
            if pid_shuffle:
                np.random.shuffle(pid_list)
            pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
            os.makedirs('partition', exist_ok=True)
            np.save(pid_fname, pid_list)
        # pid_list = np.load(pid_fname)
        pid_list = np.load(pid_fname, allow_pickle=True)
        train_list = pid_list[0]
        test_list = pid_list[1]
    else:
        pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
            dataset, pid_num, pid_shuffle))
        # print(pid_fname)
        pid_list = list()
        if not osp.exists(pid_fname):
            for i in range(1,10307):
                if i % 2!=0:
                    # counteven += 1 
                    tem = '%05d'%i
                    pid_list.append(tem)  
            for i in range(1,10307):
                if i % 2==0:
                    # countodd += 1 
                    tem = '%05d'%i
                    pid_list.append(tem)
            pid_list.append('10307')
            # pid_list = sorted(list(set(label)))
            if pid_shuffle:
                np.random.shuffle(pid_list)
            pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
            os.makedirs('partition', exist_ok=True)
            np.save(pid_fname, pid_list)
        # pid_list = np.load(pid_fname)
        pid_list = np.load(pid_fname, allow_pickle=True)
        train_list = pid_list[0]
        test_list = pid_list[1]
        # print(len(train_list))
    print('lentestdata--',len(train_list),len(test_list))
    train_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        [seq_type[i] for i, l in enumerate(label) if l in train_list],
        [view[i] for i, l in enumerate(label)
         if l in train_list],
        cache, resolution, cut=True,lmdb_path='/mnt/data/ctr/gait/OUMVLP_lmdb/',index_dict=self.index_dict)
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        cache, resolution, cut=True,lmdb_path='/mnt/data/ctr/gait/OUMVLP_lmdb/',index_dict=self.index_dict)
    print('len train,test--',len(train_source),len(test_source))
    # print(train_source[0])
    # print(test_source[0])
    return train_source, test_source
