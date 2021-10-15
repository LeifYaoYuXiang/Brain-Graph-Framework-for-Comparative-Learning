import os
import numpy as np
import torch
from pytorch_metric_learning.losses import NTXentLoss
import torch.nn.functional as F
from torch.optim import lr_scheduler
# 自定义模块
from network import GCN, GIN
from train_test import train_test_pretune
from util_deep_learning import save_sacred_metric, calculate_mean_and_standard_variance


def pretune(_run, dataloader_dir, train1_loader_type, train2_loader_type, unaug_loader_type,
            cv_number, n_epoch, config_dic,
            save_model_epoch_number, model_save_dir):
    avg_acc = []
    avg_loss = []
    avg_f1 = []

    for i in range(cv_number):
        print('第'+str(i+1)+'次训练开始')
        # 定义模型
        gcn_model = GCN(in_feats=246, n_hidden=64, n_classes=2, n_layers=5,
                        node_each_graph=246,
                        activation=F.relu, dropout=0.5)

        n_layers = config_dic['n_layers']
        n_hidden = config_dic['n_hidden']
        drop_out = config_dic['drop_out']
        pooling_type = config_dic['pooling_type']
        nt_xent_loss_temperature = config_dic['nt_xent_loss_temperature']
        adam_learning_rate = config_dic['adam_learning_rate']
        adam_weight_decay = config_dic['adam_weight_decay']
        step_size = config_dic['step_size']
        gamma = config_dic['gamma']
        alpha = config_dic['alpha']

        # gin_model = GIN(n_layers=n_layers, n_mlp_layers=2,
        #                 in_feats=246, n_hidden=n_hidden, n_classes=2,
        #                 node_each_graph=246,
        #                 final_dropout=drop_out, learn_eps=True, graph_pooling_type=pooling_type,
        #                 neighbor_pooling_type=pooling_type)
        encoder_loss_fcn = NTXentLoss(temperature=nt_xent_loss_temperature)

        encoder_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=adam_learning_rate, weight_decay=adam_weight_decay)
        # encoder_optimizer = torch.optim.Adam(gin_model.parameters(), lr=adam_learning_rate, weight_decay=adam_weight_decay)
        encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=step_size, gamma=gamma)

        train_loader1_list = []
        train_loader1_path_this_cv = os.path.join(dataloader_dir, train1_loader_type, str(i), 'train')
        for loader in os.listdir(train_loader1_path_this_cv):
            train_loader1_list.append(os.path.join(train_loader1_path_this_cv, loader))

        train_loader2_list = []
        train_loader2_path_this_cv = os.path.join(dataloader_dir, train2_loader_type, str(i), 'train')
        for loader in os.listdir(train_loader2_path_this_cv):
            train_loader2_list.append(os.path.join(train_loader2_path_this_cv, loader))

        unaug_train_list = []
        unaug_train_loader_path_this_cv = os.path.join(dataloader_dir, unaug_loader_type, str(i), 'train')
        for loader in os.listdir(unaug_train_loader_path_this_cv):
            unaug_train_list.append(os.path.join(unaug_train_loader_path_this_cv, loader))

        unaug_test_list = []
        unaug_test_loader_path_this_cv = os.path.join(dataloader_dir, unaug_loader_type, str(i), 'test')
        for loader in os.listdir(unaug_test_loader_path_this_cv):
            unaug_test_list.append(os.path.join(unaug_test_loader_path_this_cv, loader))

        encoder_loss_record, acc_record, f1_record = train_test_pretune(
            n_epoch=n_epoch,
            model=gcn_model,
            encoder_optimizer=encoder_optimizer,
            encoder_loss_fcn=encoder_loss_fcn,
            encoder_scheduler=encoder_scheduler,
            train_loader_list=train_loader1_list,
            train_loader2_list=train_loader2_list,
            unaug_train_loader_list=unaug_train_list,
            unaug_test_loader_list=unaug_test_list,
            save_model_epoch_number=save_model_epoch_number,
            model_save_dir=model_save_dir,
            cv_time=i,
            alpha=alpha
        )

        save_sacred_metric(_run, 'EncoderLoss_' + str(i+1), encoder_loss_record)
        save_sacred_metric(_run, 'Acc_'+str(i+1), acc_record)
        save_sacred_metric(_run, 'F1_'+str(i+1), f1_record)

        if i == 0:
            avg_acc = np.array(acc_record)
            avg_loss = np.array(encoder_loss_record)
            avg_f1 = np.array(f1_record)
        else:
            avg_acc = avg_acc + np.array(acc_record)
            avg_loss = avg_loss + np.array(encoder_loss_record)
            avg_f1 = avg_f1 + np.array(f1_record)

    # 保存平均值
    avg_acc = avg_acc / cv_number
    avg_acc = avg_acc.tolist()
    save_sacred_metric(_run, 'Average Accuracy', avg_acc)

    avg_loss = avg_loss / cv_number
    avg_loss = avg_loss.tolist()
    save_sacred_metric(_run, 'Average Loss', avg_loss)

    avg_f1 = avg_f1 / cv_number
    avg_f1 = avg_f1.tolist()
    save_sacred_metric(_run, 'Average F1', avg_f1)

    avg_mean, avg_std = calculate_mean_and_standard_variance(avg_acc)
    print('Average AvgAcc: ' + str(avg_mean) + '; Standard Variance AvgAcc: ' + str(avg_std))

    f1_mean, f1_std = calculate_mean_and_standard_variance(avg_f1)
    print('Average AvgF1: ' + str(f1_mean) + '; Standard Variance AvgF1: ' + str(f1_std))
