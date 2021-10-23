import os

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from comparative_exp.graph_mlp.graph_mlp_cl import GMLP_CL
from comparative_exp.graph_mlp.utils import get_A_r, Ncontrast

from data_process.data_preprocess import load_data_loaders_from_pkl
from train_test import acc_score, f1_score, precision_metric, recall_metric
from util_deep_learning import save_sacred_metric, calculate_mean_and_standard_variance


def gmlp_exp_cl(_run, dataloader_dir, loader_type,
               cv_number, n_epoch, alpha, tau, order, n_hidden, drop_out, adam_learning_rate, step_size, gamma):

    avg_acc = []
    avg_f1 = []

    for cv_time in range(cv_number):
        print('第'+str(cv_time+1)+'次对比实验开始')
        gmlp_model_cl = GMLP_CL(in_feats=246, n_hidden=n_hidden, n_classes=2,
                              node_each_graph=246, dropout=drop_out)
        cl_optimizer = torch.optim.Adam(gmlp_model_cl.parameters(), lr=adam_learning_rate)
        cl_scheduler = lr_scheduler.StepLR(cl_optimizer, step_size=step_size, gamma=gamma)

        unaug_train_list = []
        unaug_train_loader_path_this_cv = os.path.join(dataloader_dir, loader_type, str(cv_time), 'train')
        for loader in os.listdir(unaug_train_loader_path_this_cv):
            unaug_train_list.append(os.path.join(unaug_train_loader_path_this_cv, loader))

        unaug_test_list = []
        unaug_test_loader_path_this_cv = os.path.join(dataloader_dir, loader_type, str(cv_time), 'test')
        for loader in os.listdir(unaug_test_loader_path_this_cv):
            unaug_test_list.append(os.path.join(unaug_test_loader_path_this_cv, loader))

        acc_cl_exp = []
        loss_cl_exp = []
        f1_cl_exp = []

        for n in range(n_epoch):
            unaug_train_loader_path_this_epoch = unaug_train_list[n]
            unaug_test_loader_path_this_epoch = unaug_test_list[n]

            unaug_train_loader_this_epoch = load_data_loaders_from_pkl(unaug_train_loader_path_this_epoch)
            unaug_test_loader_this_epoch = load_data_loaders_from_pkl(unaug_test_loader_path_this_epoch)

            gmlp_model_cl.train()

            loss_total = 0
            # 某一个 epoch 里面的 训练部分
            for i in range(len(unaug_train_loader_this_epoch)):
                graph_info = unaug_train_loader_this_epoch[i]
                graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
                batch_size = graph_info['batch_size']
                graph = graph_info['batch_graph']
                graph = dgl.add_self_loop(graph)
                input = (graph_node_features, graph, batch_size)
                output, x_dis = gmlp_model_cl(input)
                graph_batch_label = torch.tensor(np.array(graph_info['batch_label']), dtype=torch.long)

                # 用于Abide数据集的label的修改情况
                graph_batch_label[graph_batch_label == 2] = 0

                adj = torch.zeros((246*batch_size, 246*batch_size))

                for m in range(batch_size):
                    adj[m*246:(m+1)*246, m*246:(m+1)*246] = graph_node_features[m*246:(m+1)*246]

                adj_label = get_A_r(adj, order)
                loss_train_class = F.nll_loss(output, graph_batch_label)
                # print(output.shape)
                # print(x_dis.shape)
                # print(adj_label.shape)
                loss_Ncontrast = Ncontrast(x_dis, adj_label, tau=tau)
                loss_train = loss_train_class + loss_Ncontrast * alpha
                # loss_train = loss_Ncontrast * alpha
                cl_optimizer.zero_grad()
                loss_train.backward()
                cl_optimizer.step()
                loss_total = loss_total + loss_train.item()
            encoder_loss = loss_total/len(unaug_train_loader_this_epoch)

            # 某一个 epoch 里面的 测试部分
            gmlp_model_cl.eval()
            with torch.no_grad():
                for i in range(len(unaug_test_loader_this_epoch)):
                    graph_info = unaug_test_loader_this_epoch[i]
                    graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
                    batch_size = graph_info['batch_size']
                    graph = graph_info['batch_graph']
                    graph = dgl.add_self_loop(graph)
                    input = (graph_node_features, graph, batch_size)
                    output = gmlp_model_cl(input)

                    if i == 0:
                        logits = output
                        graph_batch_label = torch.tensor(np.array(graph_info['batch_label']), dtype=torch.long)
                    else:
                        logits = torch.cat((logits, output), 0)
                        graph_batch_label = torch.cat((graph_batch_label, torch.tensor(np.array(graph_info['batch_label']), dtype=torch.long)), 0)
                _, indices = torch.max(logits, dim=1)

                # 用于Abide数据集的label的修改情况
                graph_batch_label[graph_batch_label == 2] = 0

                acc = acc_score(indices, graph_batch_label)
                f1 = f1_score(precision=precision_metric(indices, graph_batch_label), recall=recall_metric(indices, graph_batch_label))

            loss_cl_exp.append(encoder_loss)
            acc_cl_exp.append(acc)
            f1_cl_exp.append(f1)

            print("CL Epoch {:05d} | Encoder Loss {:.4f}| Accuracy {:.4f}".format(n, encoder_loss, acc))

            if n > 20:
                cl_scheduler.step()


        save_sacred_metric(_run, 'CL_GMLP_EncoderLoss_' + str(cv_time+1), loss_cl_exp)
        save_sacred_metric(_run, 'CL_GMLP_Acc_'+str(cv_time+1), acc_cl_exp)
        save_sacred_metric(_run, 'CL_GMLP_F1_'+str(cv_time+1), f1_cl_exp)

        if cv_time == 0:
            avg_acc = np.array(acc_cl_exp)
            avg_f1 = np.array(f1_cl_exp)
        else:
            avg_acc = avg_acc + np.array(acc_cl_exp)
            avg_f1 = avg_f1 + np.array(f1_cl_exp)

    avg_acc = avg_acc / cv_number
    avg_acc = avg_acc.tolist()
    save_sacred_metric(_run, 'CL_GMLP_AvgAcc', avg_acc)

    avg_f1 = avg_f1 / cv_number
    avg_f1 = avg_f1.tolist()
    save_sacred_metric(_run, 'CL_GMLP_Average_F1', avg_f1)

    arr_mean, arr_std = calculate_mean_and_standard_variance(avg_acc)
    print('CL_GMLP_Average AvgAcc: ' + str(arr_mean) + '; CL_GMLP_Standard Variance AvgAcc: ' + str(arr_std))

    f1_mean, f1_std = calculate_mean_and_standard_variance(avg_f1)
    print('CL_GMLP_Average AvgF1: ' + str(f1_mean) + '; CL_GMLP_Standard Variance AvgF1: ' + str(f1_std))