import os, json, joblib
from random import sample
import torch
import dgl
from scipy import sparse
import numpy as np
from configparser import ConfigParser
from tqdm import trange

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


config = ConfigParser()
config.read('../parameters.ini', encoding='UTF-8')


# 1: 73
# 2： 98
def nyu_data_info_cv():
    voxel_file_path = config.get('abide_path_9', 'voxel_dir')
    file_id_label_path = config.get('abide_path_9', 'file_id_label_path')

    file_name_list = os.listdir(voxel_file_path)
    with open(file_id_label_path, 'r') as load_f:
        file_id_label_dict = json.load(load_f)
    data_label_one_list = []
    data_label_two_list = []
    for item in file_name_list:
        if file_id_label_dict[item] == 1:
            data_label_one_list.append(item)
        else:
            data_label_two_list.append(item)
    return data_label_one_list, data_label_two_list


# 生成k-fold的CV
def k_fold_generate_data(k_fold_times,
                         data_label_one_list, data_label_two_list,
                         label_one_number=73, label_two_number=73):
    cv_info_json = []
    sampled_data_label_one_list = sample(data_label_one_list, label_one_number)
    sampled_data_label_two_list = sample(data_label_two_list, label_two_number)
    X = np.array(sampled_data_label_one_list + sampled_data_label_two_list)
    y = np.array([1] * label_one_number + [2] * label_two_number)
    skf = StratifiedKFold(n_splits=k_fold_times)
    for train_index, test_index in skf.split(X, y):
        cv_info_json.append({
            'train': X[train_index].tolist(),
            'test': X[test_index].tolist(),
        })
    return cv_info_json


# 加载批次数据
def load_raw_data(raw_data_txt_path_list, percent=0.1):
    graph_list = []
    for i in range(len(raw_data_txt_path_list)):
        txt_array = np.loadtxt(raw_data_txt_path_list[i])
        min_max_scaler = MinMaxScaler()
        scaled_txt_array = min_max_scaler.fit_transform(txt_array.T)
        txt_array = scaled_txt_array.T
        abs_array = abs(txt_array)

        # 生成稀疏矩阵
        baseline = np.quantile(abs_array, 1-percent)
        abs_array[abs_array < baseline] = 0
        arr_sparse = sparse.csr_matrix(abs_array)

        # 生成DGL图结构
        graph = dgl.from_scipy(arr_sparse)
        # 生成DGL图的点特征
        min_max_scaler = MinMaxScaler()

        # 按行归一化
        scaled_array = min_max_scaler.fit_transform(txt_array.T)
        array = scaled_array.T

        # 按列归一化
        # array = min_max_scaler.fit_transform(txt_array)

        graph.ndata['feat'] = torch.from_numpy(array)
        graph_list.append(graph)
    return graph_list


def generate_data_in_one_epoch(data, label, batch_size):
    batch_graph_in_one_dataloader = []
    graph_number = len(label)
    for i in range(0, graph_number, batch_size):
        start = i
        if i + batch_size < graph_number:
            end = i + batch_size
        else:
            end = graph_number
        graph_temp_list = data[start: end]

        temp_batch_size = end - start
        temp_batch_graph = dgl.batch(graph_temp_list)
        temp_batch_label = label[start: end]
        batch_graph_in_one_dataloader.append({
            'batch_graph': temp_batch_graph,
            'batch_label': temp_batch_label,
            'batch_size': temp_batch_size
        })
    return batch_graph_in_one_dataloader


def abide_data_preprocess(voxel_to_bold, bold_to_fc, base_dir,
                          maximum_epoch_number, save_dir, cv_info, percent, batch_size):
    voxel_dir_path = config.get('abide_path_9', 'voxel_dir')
    file_name_list = os.listdir(voxel_dir_path)

    file_id_label_path = config.get('abide_path_9', 'file_id_label_path')
    with open(file_id_label_path, 'r') as load_f:
        file_id_label_dict = json.load(load_f)

    data_label_one_list = []
    data_label_two_list = []

    for item in file_name_list:
        if file_id_label_dict[item] == 1:
            data_label_one_list.append(item)
        else:
            data_label_two_list.append(item)

    voxel_to_bold_to_fc = voxel_to_bold + '_' + bold_to_fc
    voxel_to_bold_to_fc_path = os.path.join(save_dir, voxel_to_bold_to_fc)
    # 创建文件夹：'F:\AbideData\dataloader\no_aug_no_aug'
    os.mkdir(voxel_to_bold_to_fc_path)

    for cv_number in range(len(cv_info)):
        voxel_to_bold_to_fc_cv_number_path = os.path.join(voxel_to_bold_to_fc_path, str(cv_number))
        # 创建文件夹：'F:\AbideData\dataloader\no_aug_no_aug\0'
        os.mkdir(voxel_to_bold_to_fc_cv_number_path)
        voxel_to_bold_to_fc_cv_number_train_path = os.path.join(voxel_to_bold_to_fc_cv_number_path, 'train')
        voxel_to_bold_to_fc_cv_number_test_path = os.path.join(voxel_to_bold_to_fc_cv_number_path, 'test')
        # 创建文件夹：'F:\AbideData\dataloader\no_aug_no_aug\0\train'
        os.mkdir(voxel_to_bold_to_fc_cv_number_train_path)
        # 创建文件夹：'F:\AbideData\dataloader\no_aug_no_aug\0\test'
        os.mkdir(voxel_to_bold_to_fc_cv_number_test_path)

        train_index = cv_info[cv_number]['train']
        test_index = cv_info[cv_number]['test']

        for n_epoch in trange(maximum_epoch_number):
            raw_train_data_txt_path_list = []
            raw_test_data_txt_path_list = []

            # 加载train的数据
            train_label = []
            for i in range(len(train_index)):
                train_filepath = base_dir + '/' + train_index[i] + '/' + voxel_to_bold_to_fc + '/' + str(n_epoch) + '.txt'
                if train_index[i] in data_label_one_list:
                    train_label.append(1)
                else:
                    train_label.append(2)
                raw_train_data_txt_path_list.append(train_filepath)
            train_data = load_raw_data(raw_train_data_txt_path_list, percent=percent)
            train_batch_graph_in_one_dataloader = generate_data_in_one_epoch(train_data, train_label, batch_size=batch_size)
            # 生成数据集
            train_pkl_path = os.path.join(voxel_to_bold_to_fc_cv_number_train_path, str(n_epoch)+'.pkl')
            with open(train_pkl_path, 'wb') as train_pkl:
                joblib.dump(train_batch_graph_in_one_dataloader, train_pkl)

            # 加载test的数据
            test_label = []
            for i in range(len(test_index)):
                test_filepath = base_dir + '/' + test_index[i] + '/' + voxel_to_bold_to_fc + '/' + str(n_epoch) + '.txt'
                if test_index[i] in data_label_one_list:
                    test_label.append(1)
                else:
                    test_label.append(2)
                raw_test_data_txt_path_list.append(test_filepath)
            test_data = load_raw_data(raw_test_data_txt_path_list, percent=percent)
            test_batch_graph_in_one_dataloader = generate_data_in_one_epoch(test_data, test_label, batch_size=batch_size)
            # 生成数据集
            test_pkl_path = os.path.join(voxel_to_bold_to_fc_cv_number_test_path, str(n_epoch)+'.pkl')
            with open(test_pkl_path, 'wb') as test_pkl:
                joblib.dump(test_batch_graph_in_one_dataloader, test_pkl)


def main():
    maximum_epoch_number = 200
    percent = 0.5
    batch_size = 8

    base_dir = config.get('abide_path_9', 'fc_matrix_dir')
    save_dir = config.get('abide_path_9', 'dataloader_dir')
    cv_info_path = config.get('abide_path_9', 'cv_info_path')
    # file_id_label_path = config.get('abide_path_9', 'file_id_label_path9)

    # # 生成相关的cv_info
    # nyu_data_label_one_list, nyu_data_label_two_list = nyu_data_info_cv()
    # cv_info = k_fold_generate_data(k_fold_times=5, data_label_one_list=nyu_data_label_one_list,
    #                                data_label_two_list=nyu_data_label_two_list,
    #                                label_one_number=73, label_two_number=73)
    # # 写入
    # json_str = json.dumps(cv_info)
    # with open(cv_info_path, 'w') as json_file:
    #     json_file.write(json_str)

    # 读取
    with open(cv_info_path, 'r') as load_f:
        cv_info = json.load(load_f)

    print('no_aug_no_aug')
    abide_data_preprocess('no_aug', 'no_aug', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('no_aug_ratio_sample')
    abide_data_preprocess('no_aug', 'ratio_sample', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('no_aug_slide_window')
    abide_data_preprocess('no_aug', 'slide_window', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('aug_no_aug')
    abide_data_preprocess('aug', 'no_aug', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('aug_ratio_sample')
    abide_data_preprocess('aug', 'ratio_sample', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)
    print('aug_slide_window')
    abide_data_preprocess('aug', 'slide_window', base_dir, maximum_epoch_number, save_dir, cv_info, percent, batch_size)


main()