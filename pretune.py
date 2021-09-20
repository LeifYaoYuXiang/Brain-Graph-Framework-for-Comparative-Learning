import os
import numpy as np
import torch
from pytorch_metric_learning.losses import NTXentLoss
import torch.nn.functional as F
from torch.optim import lr_scheduler
# 自定义模块
from network import GCN, GIN
from train_test import train_test_pretune
from util_deep_learning import save_sacred_metric


def pretune(_run, dataloader_dir, train1_loader_type, train2_loader_type, unaug_loader_type,
            cv_number, n_epoch,
            gcn_layers, n_hidden, drop_out, nt_xent_loss_temperature,
            adam_learning_rate, adam_weight_decay, step_size, gamma,
            save_model_epoch_number, model_save_dir):
    avg_acc = []

    for i in range(cv_number):
        print('第'+str(i+1)+'次训练开始')
        # 定义模型
        # gcn_model = GCN(in_feats=246, n_hidden=n_hidden, n_classes=2, n_layers=gcn_layers,
        #                 node_each_graph=246,
        #                 activation=F.relu, dropout=drop_out)
        gin_model = GIN(n_layers=5, n_mlp_layers=2,
                        in_feats=246, n_hidden=64, n_classes=2,
                        node_each_graph=246,
                        final_dropout=0.5, learn_eps=True, graph_pooling_type='sum',
                        neighbor_pooling_type='sum')
        encoder_loss_fcn = NTXentLoss(temperature=nt_xent_loss_temperature)
        # encoder_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=adam_learning_rate, weight_decay=adam_weight_decay)
        encoder_optimizer = torch.optim.Adam(gin_model.parameters(), lr=adam_learning_rate, weight_decay=adam_weight_decay)
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
            model=gin_model,
            gcn_optimizer=encoder_optimizer,
            gcn_loss_fcn=encoder_loss_fcn,
            encoder_scheduler=encoder_scheduler,
            train_loader_list=train_loader1_list,
            train_loader2_list=train_loader2_list,
            unaug_train_loader_list=unaug_train_list,
            unaug_test_loader_list=unaug_test_list,
            save_model_epoch_number=save_model_epoch_number,
            model_save_dir=model_save_dir,
            cv_time=i,
        )

        save_sacred_metric(_run, 'EncoderLoss_' + str(i+1), encoder_loss_record)
        save_sacred_metric(_run, 'Acc_'+str(i+1), acc_record)
        save_sacred_metric(_run, 'F1_'+str(i+1), f1_record)

        if i == 0:
            avg_acc = np.array(acc_record)
        else:
            avg_acc = avg_acc + np.array(acc_record)

    # 保存平均值
    avg_acc = avg_acc / cv_number
    avg_acc = avg_acc.tolist()
    save_sacred_metric(_run, 'AvgAcc', avg_acc)
