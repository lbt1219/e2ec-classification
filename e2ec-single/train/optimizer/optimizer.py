import torch


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def make_optimizer(net, cfg):
    cfg = cfg.train.optimizer    #读取cfg——base——train——optimizer优化器
    params = [] #初始化打印
    lr = cfg['lr'] #读取学习率
    weight_decay = cfg['weight_decay']

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in cfg['name']:
        optimizer = _optimizer_factory[cfg['name']](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg['name']](params, lr, momentum=0.9)
    return optimizer
