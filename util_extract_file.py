import zipfile, gzip
import os, shutil, traceback
from tqdm import trange


def list_filepath(dir_path, ext='zip'):
    filepath_list = []
    id_list = []
    for file in os.listdir(dir_path):
        if file.split('.')[-1] == ext:
            filepath_list.append(os.path.join(dir_path, file))
            id_list.append(file.split('_')[0])
    return filepath_list, id_list


def extract_file_from_zippped_file(zip_file_list, id_list, save_dir, file_name_wanted):
    for i in trange(len(zip_file_list)):
        filepath = zip_file_list[i]
        id = id_list[i]
        zip_file = zipfile.ZipFile(filepath)
        name_list = zip_file.namelist()
        for filename in name_list:
            if file_name_wanted in filename:
                zip_file.extract(filename, save_dir)
                move_file(os.path.join(save_dir, id, 'MNINonLinear\Results\\rfMRI_REST1_LR'), os.path.join(save_dir, id), file_name_wanted)
                shutil.rmtree(os.path.join(save_dir, id, 'MNINonLinear'))
                extract_file_from_gz_file(os.path.join(save_dir, id, file_name_wanted))
                os.remove(os.path.join(save_dir, id, file_name_wanted))
        zip_file.close()


def extract_file_from_gz_file(file_path):
    file_name = file_path.replace(".gz", "")
    g_file = gzip.GzipFile(file_path)
    open(file_name, "wb+").write(g_file.read())
    g_file.close()


def move_file(src_dir, dst_dir, file):
    try:
        f_src = os.path.join(src_dir, file)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        f_dst = os.path.join(dst_dir, file)
        shutil.move(f_src, f_dst)
    except Exception as e:
        print('move_file ERROR: ',e)
        traceback.print_exc()


if __name__ == '__main__':
    base_dir = 'E:\\HCP_3T_220'
    filepath_list, id_list = list_filepath(base_dir)
    extract_file_from_zippped_file(zip_file_list=filepath_list, id_list=id_list, save_dir='E:\\helloworld', file_name_wanted='rfMRI_REST1_LR.nii.gz')
