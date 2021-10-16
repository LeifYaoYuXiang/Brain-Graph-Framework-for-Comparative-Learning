import json

from configparser import ConfigParser

import joblib
import pandas as pd
import numpy as np
from random import sample
from sklearn.model_selection import StratifiedKFold
import os

from tqdm import trange

from data_process.data_preprocess import load_raw_data, generate_data_in_one_epoch

config = ConfigParser()
config.read('../parameters.ini', encoding='UTF-8')

data_description_path = config.get('abide_path', 'data_description')

# 1: 75
# 2： 100
def nyu_data_info_cv():
    extension = '_alff.nii.gz'
    nyu_data_info_cv_dict = {}
    nyu_data_label_one_list = []
    nyu_data_label_two_list = []
    data_description = pd.read_csv(data_description_path)
    for index, row in data_description.iterrows():
        if row['SITE_ID'] == 'NYU' and row['FILE_ID'] != 'no_filename':
            nyu_data_info_cv_dict[row['FILE_ID'] + extension] = row['DX_GROUP']
            if row['DX_GROUP'] == 1:
                nyu_data_label_one_list.append(row['FILE_ID'] + extension)
            else:
                nyu_data_label_two_list.append(row['FILE_ID'] + extension)
    return nyu_data_info_cv_dict, nyu_data_label_one_list, nyu_data_label_two_list


def k_fold_generate_data(k_fold_times,
                         data_label_one_list, data_label_two_list,
                         label_one_number=75, label_two_number=75):
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


def abide_data_preprocess(voxel_to_bold, bold_to_fc, base_dir,
                          maximum_epoch_number, save_dir, cv_info, percent, batch_size):
    voxel_to_bold_to_fc = voxel_to_bold + '_' + bold_to_fc
    voxel_to_bold_to_fc_path = os.path.join(save_dir, voxel_to_bold_to_fc)
    os.mkdir(voxel_to_bold_to_fc_path)
    for cv_number in range(len(cv_info)):
        voxel_to_bold_to_fc_cv_number_path = os.path.join(voxel_to_bold_to_fc_path, str(cv_number))
        os.mkdir(voxel_to_bold_to_fc_cv_number_path)

        voxel_to_bold_to_fc_cv_number_train_path = os.path.join(voxel_to_bold_to_fc_cv_number_path, 'train')
        os.mkdir(voxel_to_bold_to_fc_cv_number_train_path)
        voxel_to_bold_to_fc_cv_number_test_path = os.path.join(voxel_to_bold_to_fc_cv_number_path, 'test')
        os.mkdir(voxel_to_bold_to_fc_cv_number_test_path)

        train_index = cv_info[cv_number]['train_file_id']
        test_index = cv_info[cv_number]['test_file_id']

        for n_epoch in trange(maximum_epoch_number):

            raw_train_data_txt_path_list = []
            raw_train_label_list = []

            raw_test_data_txt_path_list = []
            raw_test_label_list = []

            # 加载train的数据
            for i in range(len(train_index)):
                train_filepath = base_dir + '/' + train_index[i] + '/' + voxel_to_bold_to_fc + '/' + str(n_epoch) + '.txt'
                raw_train_data_txt_path_list.append(train_filepath)
                raw_train_label_list.append(train_index[i])
            train_data, train_label, train_id = load_raw_data(raw_train_data_txt_path_list, raw_train_label_list, percent=percent)
            train_batch_graph_in_one_dataloader = generate_data_in_one_epoch(train_data, train_label, train_id, batch_size=batch_size)
            # 生成数据集
            train_pkl_path = os.path.join(voxel_to_bold_to_fc_cv_number_train_path, str(n_epoch)+'.pkl')
            train_pkl = open(train_pkl_path, 'wb')
            joblib.dump(train_batch_graph_in_one_dataloader, train_pkl)

            # 加载test的数据
            for i in range(len(test_index)):
                test_filepath = base_dir + '/' + test_index[i] + '/' + voxel_to_bold_to_fc + '/' + str(n_epoch) + '.txt'
                raw_test_data_txt_path_list.append(test_filepath)
                raw_test_label_list.append(test_index[i])
            test_data, test_label, test_id = load_raw_data(raw_test_data_txt_path_list, raw_test_label_list, percent=percent)
            test_batch_graph_in_one_dataloader = generate_data_in_one_epoch(test_data, test_label, test_id, batch_size=batch_size)
            # 生成数据集
            test_pkl_path = os.path.join(voxel_to_bold_to_fc_cv_number_test_path, str(n_epoch)+'.pkl')
            test_pkl = open(test_pkl_path, 'wb')
            joblib.dump(test_batch_graph_in_one_dataloader, test_pkl)


if __name__ == '__main__':
    maximum_epoch_number = 200
    percent = 0.1
    batch_size = 8

    base_dir = config.get('abide_path', 'fc_matrix_dir')
    save_dir = config.get('abide_path', 'dataloader_dir')
    cv_info_path = config.get('abide_path', 'cv_info_path')
    file_id_label_path = config.get('abide_path', 'file_id_label_path')

    # 生成相关的cv_info
    # nyu_data_info_cv_dict, nyu_data_label_one_list, nyu_data_label_two_list = nyu_data_info_cv()
    #
    # cv_info = k_fold_generate_data(k_fold_times=5, data_label_one_list=nyu_data_label_one_list,
    #                                data_label_two_list=nyu_data_label_two_list,
    #                                label_one_number=75, label_two_number=75)
    #
    # file_id_label_dict = {}
    # for nyu_data_label_one in nyu_data_label_one_list:
    #     file_id_label_dict[nyu_data_label_one] = 1
    # for nyu_data_label_two in nyu_data_label_two_list:
    #     file_id_label_dict[nyu_data_label_two] = 2
    # # 写入
    # json_str = json.dumps(cv_info)
    # with open(cv_info_path, 'w') as json_file:
    #     json_file.write(json_str)
    #
    # json_str = json.dumps(file_id_label_dict)
    # with open(file_id_label_path, 'w') as json_file:
    #     json_file.write(json_str)

    # 读取
    with open(cv_info_path, 'r') as load_f:
        cv_info = json.load(load_f)

    with open(file_id_label_path, 'r') as load_f:
        file_id_label_dict = json.load(load_f)


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


