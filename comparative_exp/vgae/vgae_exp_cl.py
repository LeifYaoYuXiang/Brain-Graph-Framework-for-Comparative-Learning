import os

import numpy as np
import torch
from torch.optim import lr_scheduler

from comparative_exp.train_test_exp import comparative_test_exp
from comparative_exp.vgae.vgae_network_cl import VGAEModel_CL
from util_deep_learning import save_sacred_metric, calculate_mean_and_standard_variance


def vgae_exp_cl(_run, dataloader_dir, loader_type,
               cv_number, n_epoch,
               adam_learning_rate, step_size, gamma):

    avg_acc = []
    avg_f1 = []

    for i in range(cv_number):
        print('第'+str(i+1)+'次对比实验开始')
        vage_model_cl = VGAEModel_CL(in_feats=246, n_hidden_1=32, n_hidden_2=16, n_classes=2, node_each_graph=246)

        cl_loss_fcn = torch.nn.CrossEntropyLoss()
        cl_optimizer = torch.optim.Adam(vage_model_cl.parameters(), lr=adam_learning_rate)
        cl_scheduler = lr_scheduler.StepLR(cl_optimizer, step_size=step_size, gamma=gamma)

        unaug_train_list = []
        unaug_train_loader_path_this_cv = os.path.join(dataloader_dir, loader_type, str(i), 'train')
        for loader in os.listdir(unaug_train_loader_path_this_cv):
            unaug_train_list.append(os.path.join(unaug_train_loader_path_this_cv, loader))

        unaug_test_list = []
        unaug_test_loader_path_this_cv = os.path.join(dataloader_dir, loader_type, str(i), 'test')
        for loader in os.listdir(unaug_test_loader_path_this_cv):
            unaug_test_list.append(os.path.join(unaug_test_loader_path_this_cv, loader))

        loss_record, acc_record, f1_record = comparative_test_exp(vage_model_cl, n_epoch=n_epoch,
                                                                  cl_loss_fcn=cl_loss_fcn, cl_optimizer=cl_optimizer, scheduler=cl_scheduler,
                                                                  train_loader_list=unaug_train_list, test_loader_list=unaug_test_list)

        save_sacred_metric(_run, 'CL_VGAE_EncoderLoss_' + str(i+1), loss_record)
        save_sacred_metric(_run, 'CL_VGAE_Acc_'+str(i+1), acc_record)
        save_sacred_metric(_run, 'CL_VGAE_F1_'+str(i+1), f1_record)

        if i == 0:
            avg_acc = np.array(acc_record)
            avg_f1 = np.array(f1_record)
        else:
            avg_acc = avg_acc + np.array(acc_record)
            avg_f1 = avg_f1 + np.array(f1_record)

    avg_acc = avg_acc / cv_number
    avg_acc = avg_acc.tolist()
    save_sacred_metric(_run, 'CL_VGAE_AvgAcc', avg_acc)

    avg_f1 = avg_f1 / cv_number
    avg_f1 = avg_f1.tolist()
    save_sacred_metric(_run, 'CL_VGAE_Average_F1', avg_f1)

    arr_mean, arr_std = calculate_mean_and_standard_variance(avg_acc)
    print('CL_VGAE_Average AvgAcc: ' + str(arr_mean) + '; CL_VGAE_Standard Variance AvgAcc: ' + str(arr_std))

    f1_mean, f1_std = calculate_mean_and_standard_variance(avg_f1)
    print('CL_VGAE_Average AvgF1: ' + str(f1_mean) + '; CL_VGAE_Standard Variance AvgF1: ' + str(f1_std))
