from configparser import ConfigParser
import pandas as pd
import numpy as np
from random import sample
from sklearn.model_selection import StratifiedKFold

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





if __name__ == '__main__':
    nyu_data_info_cv_dict, nyu_data_label_one_list, nyu_data_label_two_list = nyu_data_info_cv()
    cv_info_json = k_fold_generate_data(k_fold_times=5,
                         data_label_one_list=nyu_data_label_one_list, data_label_two_list=nyu_data_label_two_list,
                         label_one_number=75, label_two_number=75)
    print(cv_info_json)
