import os
import time

import torch
import numpy as np
import dgl

from sklearn.linear_model import LogisticRegression

from comparative_exp.graph_mlp.utils import get_A_r, Ncontrast
from data_process.data_preprocess import load_data_loaders_from_pkl
from util_deep_learning import save_model, save_net_state_dict


# 特征提取
# from util_visualization import display_matrix_2d


def train_test_pretune(
        n_epoch, model,
        encoder_optimizer, encoder_loss_fcn, encoder_scheduler,
        train_loader_list, train_loader2_list,
        unaug_train_loader_list, unaug_test_loader_list,
        save_model_epoch_number, model_save_dir, cv_time, alpha):

    dur = []
    encoder_loss_record = []
    acc_record = []
    f1_record = []
    lr_model = LogisticRegression(C=1, solver='liblinear')

    for n in range(n_epoch):
        # 训练
        t0 = time.time()

        train_loader_path_this_epoch = train_loader_list[n]
        train_loader2_path_this_epoch = train_loader2_list[n]
        unaug_train_loader_path_this_epoch = unaug_train_loader_list[n]
        unaug_test_loader_path_this_epoch = unaug_test_loader_list[n]


        train_loader_this_epoch = load_data_loaders_from_pkl(train_loader_path_this_epoch)
        train_loader2_this_epoch = load_data_loaders_from_pkl(train_loader2_path_this_epoch)
        unaug_train_loader_this_epoch = load_data_loaders_from_pkl(unaug_train_loader_path_this_epoch)
        unaug_test_loader_this_epoch = load_data_loaders_from_pkl(unaug_test_loader_path_this_epoch)
        ml_train_dataloader = unaug_train_loader_this_epoch + train_loader_this_epoch + train_loader2_this_epoch

        encoder_loss = train_encoder(model, encoder_optimizer, encoder_loss_fcn, train_loader_this_epoch, train_loader2_this_epoch, alpha)
        lr_model = train_ml_model(model, lr_model, ml_train_dataloader)
        dur.append(time.time() - t0)

        acc, f1 = test_using_ml(model, lr_model, unaug_test_loader_this_epoch)

        encoder_loss_record.append(encoder_loss)
        acc_record.append(acc)
        f1_record.append(f1)

        print("Encoder Epoch {:05d} | Time(s) {:.4f} | Encoder Loss {:.4f}| Accuracy {:.4f}".format(n, np.mean(dur), encoder_loss, acc))

        if n > 20:
            encoder_scheduler.step()

        # if n == save_model_epoch_number:
        #     save_net_state_dict(model, os.path.join(model_save_dir, str(cv_time)+'net.pkl'))
        #     save_model(lr_model, os.path.join(model_save_dir, str(cv_time)+'regression.pkl'))

    return encoder_loss_record, acc_record, f1_record


# 模型做微调
def train_test_finetune(
        n_epoch, model, lr_model,
        finetune_loss_fcn, finetune_optimizer, finetune_scheduler,
        train_loader_list, train_loader2_list,
        unaug_train_loader_list, unaug_test_loader_list):

    encoder_loss_record_finetune = []
    acc_record_finetune = []
    f1_record_fineune = []

    # 微调部分的训练与测试
    for n in range(n_epoch):
        train_loader_path_this_epoch = train_loader_list[n]
        train_loader2_path_this_epoch = train_loader2_list[n]
        unaug_train_loader_path_this_epoch = unaug_train_loader_list[n]
        unaug_test_loader_path_this_epoch = unaug_test_loader_list[n]

        train_loader_this_epoch = load_data_loaders_from_pkl(train_loader_path_this_epoch)
        train_loader2_this_epoch = load_data_loaders_from_pkl(train_loader2_path_this_epoch)
        unaug_train_loader_this_epoch = load_data_loaders_from_pkl(unaug_train_loader_path_this_epoch)
        unaug_test_loader_this_epoch = load_data_loaders_from_pkl(unaug_test_loader_path_this_epoch)

        finetune_train_dataloader = unaug_train_loader_this_epoch + train_loader_this_epoch + train_loader2_this_epoch

        encoder_loss = finetune_train(model, finetune_train_dataloader, finetune_optimizer, finetune_loss_fcn)
        acc, f1 = finetune_test(model, lr_model, unaug_test_loader_this_epoch)

        encoder_loss_record_finetune.append(encoder_loss)
        acc_record_finetune.append(acc)
        f1_record_fineune.append(f1)

        print("Finetune Epoch {:05d} | Encoder Loss {:.4f}| Accuracy {:.4f}".format(n, encoder_loss, acc))

        if n > 20:
            finetune_scheduler.step()

    return encoder_loss_record_finetune, acc_record_finetune, f1_record_fineune


# 用于训练编码器
def train_encoder(gcn_model, optimizer, loss_fcn, train_loader, train_loader2, alpha):
    # 必备，将模型设置为训练模式
    gcn_model.train()
    loss_total = 0

    for i in range(len(train_loader)):
        # 编辑输入值

        # 图一
        graph_info1 = train_loader[i]
        graph_node_features1 = graph_info1['batch_graph'].ndata['feat'].to(torch.float32)
        batch_size1 = graph_info1['batch_size']
        graph1 = graph_info1['batch_graph']
        graph1 = dgl.add_self_loop(graph1)
        input1 = (graph_node_features1, graph1, batch_size1)
        # 输出值 GIN
        # logits1, x_dis_1 = gcn_model(input1)
        # adj1 = torch.zeros((1968, 1968))
        # for m in range(batch_size1):
        #     adj1[m*246:(m+1)*246, m*246:(m+1)*246] = graph_node_features1[m*246:(m+1)*246]
        # adj_label1 = get_A_r(adj1, 2)
        # loss_Ncontrast_1 = Ncontrast(x_dis_1, adj_label1, tau=1)

        # 输出值 GCN
        logits1 = gcn_model(input1)

        # 图二
        graph_info2 = train_loader2[i]
        graph_node_features2 = graph_info2['batch_graph'].ndata['feat'].to(torch.float32)
        batch_size2 = graph_info2['batch_size']
        graph2 = graph_info2['batch_graph']
        graph2 = dgl.add_self_loop(graph2)
        input2 = (graph_node_features2, graph2, batch_size2)
        # 输出值 GIN
        # logits2, x_dis_2 = gcn_model(input2)
        # adj2 = torch.zeros((1968, 1968))
        # for m in range(batch_size1):
        #     adj2[m*246:(m+1)*246, m*246:(m+1)*246] = graph_node_features2[m*246:(m+1)*246]
        # adj_label2 = get_A_r(adj2, 2)
        # loss_Ncontrast_2 = Ncontrast(x_dis_2, adj_label2, tau=1)

        # 输出值 GCN
        logits2 = gcn_model(input2)

        # 计算损失函数
        total_node_number = logits1.size(0)
        embeddings = torch.cat((logits1, logits2))
        indices = torch.arange(total_node_number)
        label = torch.cat((indices, indices))
        # 损失值 GIN
        # loss = loss_fcn(embeddings, label) + alpha * loss_Ncontrast_1 + alpha * loss_Ncontrast_2

        # 损失值 GCN
        loss = loss_fcn(embeddings, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total = loss_total + loss.item()

    return loss_total/len(train_loader)


# 用于训练分类器（机器学习部分）
def train_ml_model(gcn_model, ml_model, unaug_train_loader):
    gcn_model.eval()
    with torch.no_grad():
        for i in range(len(unaug_train_loader)):
            graph_info = unaug_train_loader[i]
            graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
            batch_size = graph_info['batch_size']
            graph = graph_info['batch_graph']
            graph = dgl.add_self_loop(graph)
            input = (graph_node_features, graph, batch_size)
            embedding = gcn_model.get_embedding(input)
            if i == 0:
                data = embedding.numpy()
                label = np.array(graph_info['batch_label'])
            else:
                data = np.vstack((data, embedding.numpy()))
                label = np.append(label, np.array(graph_info['batch_label']))
        ml_model.fit(data, label)
    return ml_model


# 用于测试
def test_using_ml(gcn_model, ml_model, test_loader):
    gcn_model.eval()
    with torch.no_grad():
        for i in range(len(test_loader)):
            graph_info = test_loader[i]
            graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
            batch_size = graph_info['batch_size']
            graph = graph_info['batch_graph']
            graph_batch_label = torch.from_numpy(np.array(graph_info['batch_label']))
            graph = dgl.add_self_loop(graph)
            input = (graph_node_features, graph, batch_size)
            embedding = gcn_model.get_embedding(input)
            logits = torch.from_numpy(ml_model.predict(embedding.numpy()))
            if i == 0:
                indices_record = logits
                batch_y_record = graph_batch_label
            else:
                indices_record = torch.cat((indices_record, logits), 0)
                batch_y_record = torch.cat((batch_y_record, graph_batch_label), 0)
        acc = acc_score(indices_record, batch_y_record)
        f1 = f1_score(precision=precision_metric(indices_record, batch_y_record), recall=recall_metric(indices_record, batch_y_record))
        return acc, f1


# 做微调用训练
def finetune_train(model, unaug_train_loader, finetune_optimizer, finetune_loss_fcn):
    model.train()
    loss_total = 0
    for i in range(len(unaug_train_loader)):
        graph_info = unaug_train_loader[i]
        graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
        batch_size = graph_info['batch_size']
        graph = graph_info['batch_graph']
        graph = dgl.add_self_loop(graph)
        input = (graph_node_features, graph, batch_size)
        logits, x_dis = model(input)
        graph_batch_label = torch.tensor(np.array(graph_info['batch_label']), dtype=torch.long)

        loss = finetune_loss_fcn(logits, graph_batch_label)
        finetune_optimizer.zero_grad()
        loss.backward()
        finetune_optimizer.step()
        loss_total = loss_total + loss.item()
    return loss_total/len(unaug_train_loader)


# 微调之后的测试
def finetune_test(model, ml_model, unaug_test_loader):
    model.eval()
    with torch.no_grad():
        for i in range(len(unaug_test_loader)):
            graph_info = unaug_test_loader[i]
            graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
            batch_size = graph_info['batch_size']
            graph = graph_info['batch_graph']
            graph = dgl.add_self_loop(graph)
            input = (graph_node_features, graph, batch_size)
            embedding = model.get_embedding(input)
            output = torch.from_numpy(ml_model.predict(embedding.numpy()))
            if i == 0:
                indices_record = output
                batch_y_record = torch.tensor(np.array(graph_info['batch_label']), dtype=torch.long)
            else:
                indices_record = torch.cat((indices_record, output), 0)
                batch_y_record = torch.cat((batch_y_record, torch.tensor(np.array(graph_info['batch_label']), dtype=torch.long)), 0)
        acc = acc_score(indices_record, batch_y_record)
        f1 = f1_score(precision=precision_metric(indices_record, batch_y_record), recall=recall_metric(indices_record, batch_y_record))
        return acc, f1


# 计算acc的metric
def acc_score(indices_record, batch_y_record):
    correct = torch.sum(indices_record == batch_y_record)
    acc_score = correct.item() * 1.0 / len(batch_y_record)
    return acc_score


# 计算precision的metric
def precision_metric(indices_record, batch_y_record):
    a = indices_record == batch_y_record
    b = indices_record == 1
    tp = torch.sum(a & b)
    tp_fp = torch.sum(indices_record == 1)

    if tp_fp.item() * 1.0 != 0:
        return tp.item() * 1.0 / tp_fp.item() * 1.0
    else:
        return 0


# 计算recall的metric
def recall_metric(indices_record, batch_y_record):
    a = indices_record == batch_y_record
    b = batch_y_record == 1
    tp = torch.sum(a & b)
    tp_fn = torch.sum(batch_y_record == 1)
    if tp_fn.item() * 1.0 != 0:
        return tp.item() * 1.0 / tp_fn.item() * 1.0
    else:
        return 0


# 计算f1的metric
def f1_score(precision, recall):
    if precision + recall != 0:
        return (2*precision*recall) / (precision + recall)
    else:
        return 0
