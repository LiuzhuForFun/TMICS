import torch
import torch.serialization
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

import datetime
from PIL import Image
from TMICS_Heavy import TMICS
import warnings
import config as config
from multi_read_data_eval import MemoryFriendlyLoader as MemoryFriendlyLoader

warnings.filterwarnings("ignore", module="matplotlib.pyplot")
# ------------------------------
# I don't know whether you have a GPU.
plt.switch_backend('agg')
from collections import namedtuple

task = config.task  # 测试任务的类别
dataset_dir = config.dataset_dir  # 测试数据集
dataset_gtc_dir = config.dataset_gtc_dir  # 测试用到的gtc文件的路径
out_img_dir = config.out_img_dir  # 测试结果存放位置
pathlistfile = config.pathlistfile  # 具体测试用例名称表
model_path = config.model_path  # 要测试的模型位置
gpuID = config.gpuID
BATCHSIZE = config.BATCH_SIZE
h = config.h
w = config.w
N = config.N
map_location = config.map_location

if task == '':
    raise ValueError('Missing [--task].\nPlease enter the training task.')
elif task not in ['interp', 'denoise', 'denoising', 'sr', 'super-resolution']:
    raise ValueError('Invalid [--task].\nOnly support: [interp, denoise/denoising, sr/super-resolution]')

if dataset_dir == '':
    raise ValueError('Missing [--dataDir].\nPlease provide the directory of the dataset. (Vimeo-90K)')

if pathlistfile == '':
    raise ValueError('Missing [--pathlist].\nPlease provide the pathlist index file for test.')

if model_path == '':
    raise ValueError('Missing [--model model_path].\nPlease provide the path of the toflow model.')

if gpuID is None:
    cuda_flag = False
else:
    cuda_flag = True
    torch.cuda.set_device(gpuID)


# --------------------------------------------------------------

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def load_checkpoint(net, checkpoint_path, map_location):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # net.cuda_flag = checkpoint['cuda_flag']
    # net.height = checkpoint['h']
    # net.width = checkpoint['w']
    net.load_state_dict(checkpoint['net_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']
    # losses = checkpoint['losses']

    return net


def vimeo_evaluate(input_dir, dataset_gtc_dir, out_img_dir, test_codelistfile, task='', cuda_flag=True):
    mkdir_if_not_exist(out_img_dir)
    # prepare DataLoader
    Dataset = MemoryFriendlyLoader(origin_img_dir=dataset_gtc_dir, edited_img_dir=input_dir, pathlistfile=pathlistfile,
                                   task=task)
    train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
    sample_size = Dataset.count

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="./raft-small_pytorch_old.pth",
                        help="restore checkpoint, 使用旧的，不使用pytorch1.6")
    parser.add_argument('--path', default="./demo-frames", help="dataset for evaluation")
    parser.add_argument('--small', default=True, action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    print('raft is ', args.model)
    Genotype = namedtuple('Genotype', 'normal_1 normal_1_concat normal_2 normal_2_concat normal_3 normal_3_concat')

    # Make network
    video = Genotype(normal_1=[('aligned modules', 0)], normal_1_concat=[1, 2, 3],
                     normal_2=[('Residualblocks_5_1', 0), ('SPAattention_3', 1), ('ECAattention_3', 3),
                               ('Residualblocks_3_1', 4), ], normal_2_concat=[1, 2, 3, 4],
                     normal_3=[('filter of 2', 0)], normal_3_concat=[1, 2, 3])
    genotype = eval("%s" % 'video')
    model_pth = './ckpt_new_h/checkpoints_160epoch.ckpt'
    net = TMICS(h, w, args, task, cuda_flag, genotype).cuda()
    # net.load_state_dict(torch.load(model_path, map_location=map_location))
    net = load_checkpoint(net, model_path, map_location)
    # ?net.load_state_dict(torch.load(model_path))   # 报错, 主要原因是训练和测试使用的不是一个GPU
    # RuntimeError: Attempting to deserialize object on CUDA device 1 but torch.cuda.device_count() is 1
    # Please use torch.load with map_location to map your storages to an existing device.

    if cuda_flag:
        net.cuda().eval()
    else:
        net.eval()

    psnr_my = 0.0
    ssm_my = 0.0
    step = 0
    pre = 0
    processing_time_for_all = 0
    with torch.no_grad():
        for step, (x, path_code) in enumerate(train_loader):
            # x = x.cuda()    # X: 1x31xCxHxW [0:30]]共31帧
            print('x.shape: ', x.shape, x.size(1))
            for center in range(0,31):  # 使用连续3帧生成中间帧去雨的结果 [1, 2, 3, 4, 5, ..., 27, 28, 29]
                tmp = torch.zeros(size=(1, 5, 3, x.size(3), x.size(4)))  # 一次输入连续3帧
                if center ==0:
                    tmp[:, 0, :, :, :] = x[0, center, :, :, :]
                    tmp[:, 1, :, :, :] = x[0, center, :, :, :]
                    tmp[:, 2, :, :, :] = x[0, center, :, :, :]
                    # tmp[:, 2, :, :] = x[0, center, :, :, :]
                    tmp[:, 3, :, :, :] = x[0, center + 1, :, :, :]
                    tmp[:, 4, :, :, :] = x[0, center + 2, :, :, :]
                elif center ==1:
                    tmp[:, 0, :, :, :] = x[0, center-1, :, :, :]
                    tmp[:, 1, :, :, :] = x[0, center, :, :, :]
                    tmp[:, 2, :, :, :] = x[0, center, :, :, :]
                    # tmp[:, 2, :, :] = x[0, center, :, :, :]
                    tmp[:, 3, :, :, :] = x[0, center + 1, :, :, :]
                    tmp[:, 4, :, :, :] = x[0, center + 2, :, :, :]
                elif center ==29:
                    tmp[:, 0, :, :, :] = x[0, center - 2, :, :, :]
                    tmp[:, 1, :, :, :] = x[0, center - 1, :, :, :]
                    tmp[:, 2, :, :, :] = x[0, center, :, :, :]
                    # tmp[:, 2, :, :] = x[0, center, :, :, :]
                    tmp[:, 3, :, :, :] = x[0, center + 1, :, :, :]
                    tmp[:, 4, :, :, :] = x[0, center + 1, :, :, :]
                elif center ==30:
                    tmp[:, 0, :, :, :] = x[0, center - 2, :, :, :]
                    tmp[:, 1, :, :, :] = x[0, center - 1, :, :, :]
                    tmp[:, 2, :, :, :] = x[0, center, :, :, :]
                    # tmp[:, 2, :, :] = x[0, center, :, :, :]
                    tmp[:, 3, :, :, :] = x[0, center , :, :, :]
                    tmp[:, 4, :, :, :] = x[0, center , :, :, :]
                else:
                    tmp[:, 0, :, :, :] = x[0, center - 2, :, :, :]
                    tmp[:, 1, :, :, :] = x[0, center -1, :, :, :]
                    tmp[:, 2, :, :, :] = x[0, center , :, :, :]
                    # tmp[:, 2, :, :] = x[0, center, :, :, :]
                    tmp[:, 3, :, :, :] = x[0, center+1, :, :, :]
                    tmp[:, 4, :, :, :] = x[0, center+2, :, :, :]
                # tmp[:, 5, :, :, :] = x[0, center + 2, :, :, :]
                # tmp[:, 6, :, :, :] = x[0, center + 3, :, :, :]
                tmp = tmp.cuda()
                # tmp_y = torch.zeros(size=(1, 3, y.size(3), y.size(4)))
                # tmp_y[:, :, :, :, ] = y[:, index, :, :, :]
                tmp = tmp.cuda()  # 直接将x送进网络, 来进行迭代是有问题的

                predicted_img = net(tmp, iters=12, test_mode=True,opticalflow=False)  #
                img_ndarray = predicted_img.cpu().detach().numpy()
                img_ndarray = np.transpose(img_ndarray, (0, 2, 3, 1))
                img_ndarray = img_ndarray[0]
                img_tobesaved = np.asarray(img_ndarray)
                mkdir_if_not_exist(os.path.join(out_img_dir, path_code[0]))
                # video = path_code[0].split('/')[0]  # print(path_code)    # ('00001/6',)
                # sep = path_code[0].split('/')[1]
                # plt.imsave(os.path.join(out_img_dir, path_code[0], '%d.jpg' % (center+1)), np.clip(img_ndarray, 0.0, 1.0))
                plt.imsave(os.path.join(out_img_dir, path_code[0], 'result-d-07-%d.jpg' % (center + 1)),
                           np.clip(img_tobesaved, 0.0, 1.0))

        print('*' * 40)
        print('END')

vimeo_evaluate(dataset_dir, dataset_gtc_dir, out_img_dir, pathlistfile, task=task, cuda_flag=cuda_flag)
