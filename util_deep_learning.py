import os, random
import numpy as np
import torch
import joblib


# 控制深度学习的Seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# 将深度学习网络予以保存
def save_net(net, save_path):
    torch.save(net, save_path)


# 读取深度学习网络
def load_net(load_path):
    net = torch.load(load_path)
    return net


# 将深度学习网络予以保存(只保存参数而非整个模型)
def save_net_state_dict(net, save_path):
    torch.save(net.state_dict(), save_path)


# 将深度学习网络予以读取(只保存参数而非整个模型)
def load_net_state_dict(model, load_path):
    model.load_state_dict(torch.load(load_path))
    return model


# 保存机器学习的模型
def save_model(model, save_path):
    joblib.dump(model, save_path)


# 读取机器学习的模型
def load_model(save_path):
    model = joblib.load(save_path)
    return model


# 利用sacred包将metric进行存储
def save_sacred_metric(_run, metric_name, data_list):
    for j in range(len(data_list)):
        _run.log_scalar(metric_name, data_list[j], j)


# 利用


# Early Stopping: 防止过拟合
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), 'es_checkpoint.pt')