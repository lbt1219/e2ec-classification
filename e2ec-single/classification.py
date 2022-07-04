
from pathlib import Path
from network import make_network
import tqdm
import torch
import os
import nms
import post_process
from dataset.data_loader import make_demo_loader
from train.model_utils.utils import load_network
import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import imgviz
import cv2
from model import convnext_tiny as create_model
from torchvision import transforms
from PIL import Image
import json
from io import BytesIO
import PIL

class_name= ['damper',]
"""
class_name= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
"""

parser = argparse.ArgumentParser()

parser.add_argument("--config_file", default='coco',help='/path/to/config_file.py')
parser.add_argument("--image_dir",  default='/home/deep/DATA/data/anti-vibration_hammer/images',help='/path/to/images')#/home/deep/DATA/data/anti-vibration_hammer/images
parser.add_argument("--checkpoint", default='/home/deep/DATA/train_log/e2ec/real/bigdivide/steplr/sgd/159.pth', help='/path/to/model_weight.pth')
parser.add_argument("--ct_score", default=0.3, help='threshold to filter instances', type=float)
parser.add_argument("--with_nms", default=False, type=bool,
                    help='if True, will use nms post-process operation', choices=[True, False])
parser.add_argument("--with_post_process", default=True, type=bool,
                    help='if True, Will filter out some jaggies', choices=[True, False])
parser.add_argument("--stage", default='final-dml', help='which stage of the contour will be generated',
                    choices=['init', 'coarse', 'final', 'final-dml'])
parser.add_argument("--output_dir", default='/home/deep/DATA/result/damper/e2ec/real/扩充的结果/class/', help='/path/to/output_dir')#/home/deep/DATA/result/e2ec/1/
parser.add_argument("--device", default=3, type=int, help='device idx')
parser.add_argument("--vis" , default=True, type=bool, help='Whether to view the class and score')

args = parser.parse_args()

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file).config
    cfg.test.with_nms = bool(args.with_nms)
    cfg.test.test_stage = args.stage
    cfg.test.ct_score = args.ct_score
    return cfg

def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]

def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img




class Visualizer(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def classification(self,output, batch):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        num_classes = 2
        img_size = 100

        data_transform = transforms.Compose(
            [transforms.Resize(int(img_size * 1.14)),
             transforms.CenterCrop(img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # load image
        """
        img =  (dst * 255).astype(np.uint8)
        img = Image.fromarray(np.uint8(img))
        img= Image.fromarray(np.uint8(img)).convert('RGB')
        """
        dataPIL = Visualizer.visualize_ex_original(self,output, batch)
        print(type(dataPIL))
        img = data_transform(dataPIL)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)
        # class_indict = ['normal','rust']

        # create model
        model = create_model(num_classes=num_classes).to(device)
        # load model weights
        model_weight_path = "/home/deep/1-POJECT/image-progress/deep-learning-for-image-processing/pytorch_classification/ConvNeXt/weights/best_model.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        predict[predict_cla] = predict[predict_cla] * 100
        print_res = "{}   {:.6}".format(class_indict[str(predict_cla)],
                                        predict[predict_cla].numpy())
        print(print_res + '%')
        """
        for i in range(len(predict)):
            predict[i]=predict[i]*100
            print("{:10} {:10.4}".format(class_indict[str(i)],
                                                    predict[i].numpy())+'%')
            return ("{:10} {:10.4}".format(class_indict[str(i)],
                                                    predict[i].numpy())+'%')
        """
        return print_res

    def visualize_ex_original(self,output, batch, save_dir=None,):

        inp = bgr_to_rgb(unnormalize_img(batch['inp'][0], self.cfg.data.mean,
                                         self.cfg.data.std).permute(1, 2, 0))

        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy()


        fig, ax = plt.subplots(1, figsize=(3.84, 3.84),facecolor='black')
        fig.tight_layout()
        ax.axis('off')

        inp = np.array(inp)
        for i in range(len(ex)):
            mask = np.zeros((inp.shape[0],inp.shape[1]), np.uint8)
            poly = ex[i]
            cv2.polylines(mask, np.int32([poly]), 1, (255, 255, 255))  # 描绘边缘
            cv2.fillPoly(mask, np.int32([poly]), (255, 255, 255))

            dst = cv2.bitwise_and(inp, inp, mask=mask)

            x_min = int(min(poly[:, 0]))
            y_min = int(min(poly[:, 1]))
            x_max = int(max(poly[:, 0]))
            y_max = int(max(poly[:, 1]))
            dst = dst[y_min:y_max , x_min:x_max]

            ax.imshow(dst)
            buffer_ = BytesIO()  # using buffer,great way!
            # 保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
            plt.savefig(buffer_, format='jpg')
            buffer_.seek(0)
            # 用PIL或CV2从内存中读取
            dataPIL = PIL.Image.open(buffer_)
            plt.close()
            return dataPIL


            #if save_dir is not None:
                #cv2.imwrite(args.output_dir+Path(save_dir).stem+'_{}.jpg'.format(i) , dst)
                #plt.savefig(fname=args.output_dir+Path(save_dir).stem+'_{}.jpg'.format(i))#bbox_inches='tight',pad_inches=-0.05
            #else:
                #plt.show()
        #plt.close()



    def visualize_ex(self, output, batch, score , class_num, save_dir=None,):
        inp = bgr_to_rgb(unnormalize_img(batch['inp'][0], self.cfg.data.mean,
                                         self.cfg.data.std).permute(1, 2, 0))
        ex = output['py']
        #print(ex)
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(20, 10),facecolor='black')
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],#蓝
            [255, 127, 14],#橘黄
            [46, 160, 44],#绿
            [214, 40, 39],#红
            #[148, 103, 189],#棕
            #[140, 86, 75],#紫
            #[227, 119, 194],#粉
            #[126, 126, 126],#浅棕
            #[188, 189, 32],#屎黄
            #[26, 190, 207]#绿蓝
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)

        cla_res = Visualizer.classification(self,output, batch)

        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            #画点
            #ax.scatter(poly[:, 0], poly[:, 1],s=10)
            #画线
            ax.plot(poly[:, 0], poly[:, 1], color=color, lw=1)
            #画面
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.3)
            #加分数和种类

            x_min = int(min(poly[:, 0]))
            y_min = int(min(poly[:, 1]))

            text = class_name[int(class_num[0][i].item())] +"  "+str(round(score[0][i].item()*100,2))+"%"+ "\n"+cla_res+'%'
            ax.text(x_min,y_min-10, text, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k',lw=1 ,alpha=0.5))

        if save_dir is not None:
            plt.savefig(fname=save_dir, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize(self, output, batch, score=None ,class_num=None):
        if args.output_dir != 'None':
            file_name = os.path.join(args.output_dir, batch['meta']['img_name'][0])
        else:
            file_name = None
        if args.vis:
            self.visualize_ex(output, batch, score ,class_num, save_dir=file_name,)
        else :
            self.visualize_ex_original(output, batch, save_dir=file_name, )

def run_visualize(cfg):
    network = make_network.get_network(cfg).cuda()
    load_network(network, args.checkpoint)
    network.eval()

    data_loader = make_demo_loader(args.image_dir, cfg=cfg)
    visualizer = Visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        if args.vis:
            vis = {'vis': ['']}
            batch['meta'].update(vis)
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            if args.vis:
                output, score ,class_num = network(batch['inp'], batch)
            else:
                output = network(batch['inp'], batch)
        if args.with_post_process:
            post_process.post_process(output)
        if args.with_nms:
            nms.post_process(output)
        if args.vis:
            visualizer.visualize(output, batch, score ,class_num)
        else:
            visualizer.visualize(output, batch)


if __name__ == "__main__":
    cfg = get_cfg(args)
    torch.cuda.set_device(args.device)
    run_visualize(cfg)
