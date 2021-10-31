import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_all_data_path(dir):
    all_available_filepath = []
    all_available_dir = os.listdir(dir)
    for each_dir in all_available_dir:
        fc_matrix_filepath = dir + '/' + each_dir + '/' + 'no_aug_no_aug' + '/' + '0.txt'
        all_available_filepath.append(fc_matrix_filepath)
    return all_available_filepath


def get_available_data(dir, all_available_filepath):
    all_available_dir = os.listdir(dir)
    available_data = {}
    for i in range(len(all_available_filepath)):
        each_available_path = all_available_filepath[i]
        each_fc_matrix = np.triu(np.loadtxt(each_available_path),1).flatten()
        available_data[all_available_dir[i]] = each_fc_matrix
    return available_data


def train_logistic_regression(available_data):
    with open('F:\\AbideData\\cv_info.json', 'r') as load_f:
        cv_info = json.load(load_f)

    with open('F:\\AbideData\\file_id_label.json', 'r') as load_f:
        label_info = json.load(load_f)

    for i in range(len(cv_info)):
        train_index = cv_info[i]['train']
        test_index = cv_info[i]['test']

        for j in range(len(train_index)):
            if j == 0:
                X = available_data[train_index[j]]
                y = [label_info[train_index[j]]]
            else:
                X = np.vstack((X, available_data[train_index[j]]))
                y = y + [label_info[train_index[j]]]
        clf = LogisticRegression(random_state=0).fit(X, y)

        for j in range(len(test_index)):
            if j == 0:
                X = available_data[test_index[j]]
                y_true = [label_info[test_index[j]]]
            else:
                X = np.vstack((X, available_data[test_index[j]]))
                y_true = y_true + [label_info[test_index[j]]]
        y_pred = clf.predict(X)
        print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    dir = 'F:\\AbideData\\fc_matrix'
    all_available_filepath = load_all_data_path(dir)
    print('数据路径整理完成')

    available_data = get_available_data(dir, all_available_filepath)
    print('数据整理完成')
    train_logistic_regression(available_data)
