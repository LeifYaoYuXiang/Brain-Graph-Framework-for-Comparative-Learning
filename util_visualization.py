import os
from configparser import ConfigParser
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def display_matrix_2d(mat):
    plt.matshow(mat)
    plt.show()


def display_matrix_3d(mat):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    w = [i+1 for i in range(mat.shape[0])]
    q = [i+1 for i in range(mat.shape[1])]
    W,Q = np.meshgrid(w,q)

    ax.set_xlabel('x_metric')
    ax.set_ylabel('y_metric')
    ax.set_zlabel('value')
    surf = ax.plot_surface(W, Q, mat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


config = ConfigParser()
config.read('parameters.ini', encoding='UTF-8')
fc_matrix = config.get('filepath', 'fc_matrix_dir')
subject_id = '6001'
number = '0.txt'

type = 'no_aug_no_aug'
mat1 = np.loadtxt(os.path.join(fc_matrix, subject_id, type, number))
display_matrix_3d(mat1)

subject_id = '6392'
mat2 = np.loadtxt(os.path.join(fc_matrix, subject_id, type, number))
display_matrix_3d(mat2)

mat_minus = abs(mat1-mat2)
mat_minus[mat_minus < 1] = 0
display_matrix_3d(mat_minus)
