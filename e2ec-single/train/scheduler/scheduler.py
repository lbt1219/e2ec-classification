from torch.optim.lr_scheduler import MultiStepLR
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR,OneCycleLR

def make_lr_scheduler_cos(optimizer, config):
    '''
    scheduler = MultiStepLR(optimizer, milestones=config.train.optimizer['milestones'],
                            gamma=config.train.optimizer['gamma'])
    '''
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=3,eta_min=0.0001)
    return scheduler

def make_lr_scheduler_step(optimizer, config):
    scheduler = MultiStepLR(optimizer, milestones=config.train.optimizer['milestones'],
                            gamma=config.train.optimizer['gamma'])
    return scheduler


def make_lr_scheduler_cyc(optimizer, config):
    """　
    max_lr：最大学习率
　　total_steps：迭代次数
　　pct_start：学习率上升部分占比
　　div_factor：初始学习率= max_lr / div_factor
　　final_div_factor：最终学习率= 初始学习率 / final_div_factor
    """
    scheduler = OneCycleLR(optimizer, max_lr=0.0001, total_steps=config.train.epoch,
                           pct_start=0.1,div_factor=1,final_div_factor=10)
    return scheduler


def set_lr_scheduler(scheduler, config):
    scheduler.milestones = Counter(config.train.optimizer['milestones'])
    scheduler.gamma = config.train.optimizer['gamma']

