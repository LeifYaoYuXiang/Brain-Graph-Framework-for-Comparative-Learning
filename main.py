# 引入基础的包
import os
from configparser import ConfigParser
# sacred 实验数据保存
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

# 自定义函数
from comparative_exp.gae.gae_exp_cl import gae_exp_cl
from comparative_exp.gat.gat_exp_cl import gat_exp_cl
from comparative_exp.gin.gin_exp_cl import gin_exp_cl
from comparative_exp.graph_mlp.graph_mlp_exp_cl import gmlp_exp_cl
from comparative_exp.vgae.vgae_exp_cl import vgae_exp_cl
from finetune import finetune
from pretune import pretune
from util_deep_learning import seed_torch
from comparative_exp.gcn.gcn_exp_cl import gcn_exp_cl

# Omniboard 安装手册： https://shenxiaohai.me/2019/01/17/sacred-tool/#MongoDB
# 1. 首先使用node.js开启 Omniboard：```omniboard -m localhost:27017:sacred```
# 2. 在浏览器上： http://localhost:9000/sacred
# 3. 在文件夹中， 运行： python main.py
# 你也可以在main函数中调用这个方法， 例如：
# if __name__ == '__main__':
#     ex.run()
# 运行前测试参数，可以尝试：
# python main.py print_config
# 为了控制随机性，可以：
# python main.py with seed=X


os.environ["GIT_PYTHON_REFRESH"] = "quiet"
ex = Experiment("Augmented Comparative Learning With Finetune")
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='sacred'))
ex.captured_out_filter = apply_backspaces_and_linefeeds


# 超参数设置
@ex.config
def hyper_parameters_config():
    n_layers = 5
    drop_out = 0.5
    nt_xent_loss_temperature = 0.1
    adam_learning_rate = 1e-3
    adam_weight_decay = 5e-4
    n_epoch = 80
    n_hidden = 64
    step_size = 10
    gamma = 0.5
    alpha = 0
    seed = 1029
    save_model_epoch_number = 79
    pooling_type = 'sum'


# Entry
@ex.automain
def main(_run, n_layers, n_hidden, drop_out, nt_xent_loss_temperature,
         adam_learning_rate, adam_weight_decay,
         n_epoch,
         step_size, gamma, alpha, pooling_type,
         seed, save_model_epoch_number):

    config = ConfigParser()
    config.read('parameters.ini', encoding='UTF-8')
    # 设置随机数种子
    seed_torch(seed)

    cv_number = config.getint('experiment', 'cv_number')
    dataloader_dir = config.get('filepath', 'dataloader_dir')
    model_save_dir = config.get('filepath', 'model_save_dir')

    voxel_to_bold_options = ['no_aug', 'aug']
    bold_to_fc_options = ['no_aug', 'slide_window', 'ratio_sample']

    train1_loader_type = voxel_to_bold_options[1] + '_' + bold_to_fc_options[1]  # train 1: aug_slide_window
    train2_loader_type = voxel_to_bold_options[0] + '_' + bold_to_fc_options[2]  # train 2: no_aug_ratio_sample
    unaug_loader_type = voxel_to_bold_options[0] + '_' + bold_to_fc_options[0]

    gcin_config_dic = {
        'n_layers': n_layers,
        'n_hidden': n_hidden,
        'drop_out': drop_out,
        'nt_xent_loss_temperature': nt_xent_loss_temperature,
        'alpha': alpha,
        'adam_learning_rate': adam_learning_rate,
        'adam_weight_decay': adam_weight_decay,
        'step_size': step_size,
        'gamma': gamma,
        'pooling_type': pooling_type,
    }
    # # 特征提取
    pretune(_run, dataloader_dir, train1_loader_type, train2_loader_type, unaug_loader_type,
             cv_number, n_epoch, gcin_config_dic,
             save_model_epoch_number, model_save_dir)

    # 微调
    # finetune(_run, dataloader_dir, train1_loader_type, train2_loader_type, unaug_loader_type,
    #          cv_number, n_epoch,
    #          adam_learning_rate, adam_weight_decay, step_size, gamma,
    #          model_save_dir)

    #
    # # 对比试验: GCN
    # gcn_exp_cl(_run, dataloader_dir, unaug_loader_type,
    #            cv_number, n_epoch,
    #            gcn_layers, n_hidden, drop_out, adam_learning_rate, step_size, gamma)

    # 对比实验: GAT
    # gat_exp_cl(_run, dataloader_dir, unaug_loader_type,
    #            cv_number, n_epoch,
    #            gat_layers=5, n_hidden=8, drop_out=drop_out,
    #            num_heads=8, num_out_heads=1, attn_drop=0.6, in_drop=0.6, residual=False, negative_slope=0.2,
    #            adam_learning_rate=adam_learning_rate, step_size=step_size, gamma=gamma)

    # #对比试验: GIN
    # gin_exp_cl(_run, dataloader_dir, unaug_loader_type,
    #            cv_number, n_epoch,
    #            adam_learning_rate=adam_learning_rate, step_size=step_size, gamma=gamma)
    #
    # # 对比试验： Graph-MLP
    # gmlp_exp_cl(_run, dataloader_dir, unaug_loader_type,
    #              cv_number, n_epoch, alpha=5, tau=1.0, order=2, n_hidden=64, drop_out=0.6, adam_learning_rate=adam_learning_rate,
    #              step_size=step_size, gamma=gamma)
    #
    # # 对比试验： VGAE-FCNN
    # vgae_exp_cl(_run, dataloader_dir, unaug_loader_type,
    #            cv_number, n_epoch,
    #            adam_learning_rate=adam_learning_rate, step_size=step_size, gamma=gamma)
    #
    # # 对比实验: GAE
    # gae_exp_cl(_run, dataloader_dir, unaug_loader_type,
    #            cv_number, n_epoch,
    #            adam_learning_rate=adam_learning_rate, step_size=step_size, gamma=gamma)

    # # 对比试验： GraphCL

