import os
from configparser import ConfigParser
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from data_process.data_augment import load_nifti, voxel_to_bold


def display_matrix_2d(mat):
    plt.matshow(mat, vmin=0, vmax=99)
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


def display_time_series(mask_file_path, voxel_filepath):
    mask = load_nifti(mask_file_path)
    original_data = load_nifti(voxel_filepath)
    bold_matrix = voxel_to_bold(file_name='draw', original_data=original_data, mask=mask, roi_ratio=1)
    average_bold_time_series = np.mean(bold_matrix, axis=0)
    average_bold_time_series = average_bold_time_series[0].tolist()[0]
    x = range(len(average_bold_time_series))
    plt.plot(x, average_bold_time_series, 'r--')

    plt.title('BOLD Time Series')
    plt.show()
    # print(average_bold_time_series.shape)


# config = ConfigParser()
# config.read('parameters.ini', encoding='UTF-8')
# fc_matrix = config.get('filepath', 'fc_matrix_dir')
# subject_id = '6061'
# number = '0.txt'
# type = 'no_aug_no_aug'
# mat1 = np.loadtxt(os.path.join(fc_matrix, subject_id, type, number))
# # display_matrix_3d(mat1)

# if __name__ == '__main__':
#     display_time_series(mask_file_path='F:\\ComparativeLearning\\BN_Atlas_246_3mm.nii', voxel_filepath='F:\\AbideData\\func_preproc\\NYU_0050954_func_preproc.nii.gz')
