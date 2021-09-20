import os

import numpy as np
import torch
from torch.optim import lr_scheduler

from network import GIN
from train_test import train_test_finetune
from util_deep_learning import save_sacred_metric, load_model, load_net_state_dict


def finetune(_run, dataloader_dir, train1_loader_type, train2_loader_type, unaug_loader_type,
             cv_number, n_epoch,
             adam_learning_rate, adam_weight_decay, step_size, gamma,
             model_save_dir):

    avg_acc = []
    for i in range(cv_number):
        print('第'+str(i+1)+'次微调开始')
        # 定义模型
        # gcn_model = load_net(os.path.join(model_save_dir, str(i)+'net.pkl'))
        gin_model = GIN(n_layers=5, n_mlp_layers=2,
                        in_feats=246, n_hidden=64, n_classes=2,
                        node_each_graph=246,
                        final_dropout=0.5, learn_eps=True, graph_pooling_type='sum',
                        neighbor_pooling_type='sum')
        gin_model = load_net_state_dict(gin_model, os.path.join(model_save_dir, str(i)+'net.pkl'))
        lr_model = load_model(os.path.join(model_save_dir, str(i)+'regression.pkl'))

        finetune_loss_fcn = torch.nn.CrossEntropyLoss()
        finetune_optimizer = torch.optim.Adam(gin_model.parameters(), lr=adam_learning_rate, weight_decay=adam_weight_decay)
        finetune_scheduler = lr_scheduler.StepLR(finetune_optimizer, step_size=step_size, gamma=gamma)

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

        encoder_loss_record, acc_record, f1_record = train_test_finetune(
            n_epoch=n_epoch,
            model=gin_model,
            lr_model=lr_model,
            finetune_loss_fcn=finetune_loss_fcn,
            finetune_optimizer=finetune_optimizer,
            finetune_scheduler=finetune_scheduler,
            train_loader_list=train_loader1_list,
            train_loader2_list=train_loader2_list,
            unaug_train_loader_list=unaug_train_list,
            unaug_test_loader_list=unaug_test_list,
        )

        save_sacred_metric(_run, 'FineTune_EncoderLoss_' + str(i+1), encoder_loss_record)
        save_sacred_metric(_run, 'FineTune_Acc_'+str(i+1), acc_record)
        save_sacred_metric(_run, 'FineTune_F1_'+str(i+1), f1_record)

        if i == 0:
            avg_acc = np.array(acc_record)
        else:
            avg_acc = avg_acc + np.array(acc_record)

    # 保存平均值
    avg_acc = avg_acc / cv_number
    avg_acc = avg_acc.tolist()
    save_sacred_metric(_run, 'FineTune_AvgAcc', avg_acc)