from comparative_exp.gae.gae_exp_cl import gae_exp_cl
from comparative_exp.gat.gat_exp_cl import gat_exp_cl
from comparative_exp.gin.gin_exp_cl import gin_exp_cl
from comparative_exp.graph_mlp.graph_mlp_exp_cl import gmlp_exp_cl
from comparative_exp.vgae.vgae_exp_cl import vgae_exp_cl
from comparative_exp.gcn.gcn_exp_cl import gcn_exp_cl


def comparative_test(_run, dataloader_dir, unaug_loader_type,
                    cv_number, n_epoch, config_dic):

    adam_learning_rate = config_dic['adam_learning_rate']
    gamma = config_dic['gamma']
    step_size = config_dic['step_size']
    drop_out = config_dic['drop_out']

    # 对比试验: GCN
    gcn_exp_cl(_run, dataloader_dir, unaug_loader_type,
               cv_number, n_epoch,
               gcn_layers=5, n_hidden=64, drop_out=drop_out, adam_learning_rate=adam_learning_rate,
               step_size=step_size, gamma=gamma)

    # 对比实验: GAT
    gat_exp_cl(_run, dataloader_dir, unaug_loader_type,
               cv_number, n_epoch,
               gat_layers=5, n_hidden=8, drop_out=drop_out,
               num_heads=8, num_out_heads=1, attn_drop=0.6, in_drop=0.6, residual=False, negative_slope=0.2,
               adam_learning_rate=adam_learning_rate, step_size=step_size, gamma=gamma)

    #对比试验: GIN
    gin_exp_cl(_run, dataloader_dir, unaug_loader_type,
               cv_number, n_epoch,
               adam_learning_rate=adam_learning_rate, step_size=step_size, gamma=gamma)


    # 对比试验： Graph-MLP
    gmlp_exp_cl(_run, dataloader_dir, unaug_loader_type,
                 cv_number, n_epoch, alpha=5, tau=1.0, order=2, n_hidden=64, drop_out=0.6, adam_learning_rate=adam_learning_rate,
                 step_size=step_size, gamma=gamma)

    # 对比试验： VGAE-FCNN
    vgae_exp_cl(_run, dataloader_dir, unaug_loader_type,
               cv_number, n_epoch,
               adam_learning_rate=adam_learning_rate, step_size=step_size, gamma=gamma)

    # 对比实验: GAE
    gae_exp_cl(_run, dataloader_dir, unaug_loader_type,
               cv_number, n_epoch,
               adam_learning_rate=adam_learning_rate, step_size=step_size, gamma=gamma)