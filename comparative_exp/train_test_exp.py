import dgl
import numpy as np
import torch
from data_process.data_preprocess import load_data_loaders_from_pkl
from train_test import acc_score, f1_score, precision_metric, recall_metric


def comparative_test_exp(model, n_epoch, cl_loss_fcn, cl_optimizer,train_loader_list, test_loader_list, scheduler):
    acc_cl_exp = []
    loss_cl_exp = []
    f1_cl_exp = []
    for n in range(n_epoch):
        unaug_train_loader_path_this_epoch = train_loader_list[n]
        unaug_test_loader_path_this_epoch = test_loader_list[n]

        unaug_train_loader_this_epoch = load_data_loaders_from_pkl(unaug_train_loader_path_this_epoch)
        unaug_test_loader_this_epoch = load_data_loaders_from_pkl(unaug_test_loader_path_this_epoch)

        encoder_loss, acc, f1 = train_test_cl_in_one_epoch(model=model,
                                                        train_loader_this_epoch=unaug_train_loader_this_epoch,
                                                        test_loader=unaug_test_loader_this_epoch,
                                                        cl_optimizer=cl_optimizer, cl_loss_fcn=cl_loss_fcn)
        loss_cl_exp.append(encoder_loss)
        acc_cl_exp.append(acc)
        f1_cl_exp.append(f1)

        print("CL Epoch {:05d} | Encoder Loss {:.4f}| Accuracy {:.4f}".format(n, encoder_loss, acc))

        if n > 20:
            scheduler.step()
    return loss_cl_exp, acc_cl_exp, f1_cl_exp


def train_test_cl_in_one_epoch(model, train_loader_this_epoch, test_loader, cl_optimizer, cl_loss_fcn):
    model.train()
    loss_total = 0

    # 某一个 epoch 里面的 训练部分
    for i in range(len(train_loader_this_epoch)):
        graph_info = train_loader_this_epoch[i]
        graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
        batch_size = graph_info['batch_size']
        graph = graph_info['batch_graph']
        graph = dgl.add_self_loop(graph)
        input = (graph_node_features, graph, batch_size)
        logits = model(input)
        graph_batch_label = torch.tensor(np.array(graph_info['batch_label']), dtype=torch.long)

        loss = cl_loss_fcn(logits, graph_batch_label)
        cl_optimizer.zero_grad()
        loss.backward()
        cl_optimizer.step()
        loss_total = loss_total + loss.item()
    encoder_loss = loss_total/len(train_loader_this_epoch)

    # 某一个 epoch 里面的 测试部分
    model.eval()
    with torch.no_grad():
        for i in range(len(test_loader)):
            graph_info = test_loader[i]
            graph_node_features = graph_info['batch_graph'].ndata['feat'].to(torch.float32)
            batch_size = graph_info['batch_size']
            graph = graph_info['batch_graph']
            graph = dgl.add_self_loop(graph)
            input = (graph_node_features, graph, batch_size)
            output = model(input)
            if i == 0:
                logits = output
                graph_batch_label = torch.tensor(np.array(graph_info['batch_label']), dtype=torch.long)
            else:
                logits = torch.cat((logits, output), 0)
                graph_batch_label = torch.cat((graph_batch_label, torch.tensor(np.array(graph_info['batch_label']), dtype=torch.long)), 0)
        _, indices = torch.max(logits, dim=1)
        acc = acc_score(indices, graph_batch_label)
        f1 = f1_score(precision=precision_metric(indices, graph_batch_label), recall=recall_metric(indices, graph_batch_label))
    return encoder_loss, acc, f1
