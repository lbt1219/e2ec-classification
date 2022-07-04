from .base import commen, data, model, train, test
import numpy as np
data.scale = None

model.heads['ct_hm'] = 1   #种类数

train.batch_size = 12     #batch——size
train.epoch = 200     #轮数
train.dataset = 'coco_train'

test.dataset = 'coco_val'


class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
