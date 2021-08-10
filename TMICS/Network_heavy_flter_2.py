import math
import torch
# import torch.utils.serialization
# import torchfile
import torch.serialization
import torch.nn.functional as F
import torch
import torch.nn as nn
from guided_filter_pytorch.guided_filter import GuidedFilter
from torch.autograd import Variable
import sys
import matplotlib.pyplot as plt
import numpy as np
import config
from operations_m import *

sys.path.append('core')     # 注意一下
from raft import RAFT       # 没有问题
# from toolbox.arch import Feature_CARB_mutil
# from utils import utils
# from pwc.flow_pwc import Flow_PWC
# from toolbox.arch import Feature_CARB
# from toolbox.unet_ywh_multi import U_Net
# from lib.FJDN import FJDN
# from pietorch.DuRN_S import cleaner
# from pietorch.DuRN_U import cleaner

arguments_strModel = 'sintel-final'
SpyNet_model_dir = './models'  # SpyNet模型参数目录


def normalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] - 0.485) / 0.229
    tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
    tensorBlue = (tensorInput[:, 2:3, :, :] - 0.406) / 0.225
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


def denormalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] * 0.229) + 0.485
    tensorGreen = (tensorInput[:, 1:2, :, :] * 0.224) + 0.456
    tensorBlue = (tensorInput[:, 2:3, :, :] * 0.225) + 0.406
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


Backward_tensorGrid = {}


def Backward(tensorInput, tensorFlow, cuda_flag):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        if cuda_flag:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()
        else:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1)
    # end

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tensorInput,
                                           grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2,
                                                                                                                   3,
                                                                                                                   1),
                                           mode='bilinear', padding_mode='border')


class SpyNet(torch.nn.Module):
    def __init__(self, cuda_flag):
        super(SpyNet, self).__init__()
        self.cuda_flag = cuda_flag

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            # end

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)

        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(4)])

        self.load_state_dict(torch.load(SpyNet_model_dir + '/network-' + arguments_strModel + '.pytorch'), strict=False)

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = [tensorFirst]
        tensorSecond = [tensorSecond]

        for intLevel in range(3):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))

        tensorFlow = tensorFirst[0].new_zeros(tensorFirst[0].size(0), 2,
                                              int(math.floor(tensorFirst[0].size(2) / 2.0)),
                                              int(math.floor(tensorFirst[0].size(3) / 2.0)))

        for intLevel in range(len(tensorFirst)):
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear',
                                                              align_corners=True) * 2.0

            # if the sizes of upsampling and downsampling are not the same, apply zero-padding.
            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 1, 0, 0], mode='replicate')

            # input ：[first picture of corresponding level,
            # 		   the output of w with input second picture of corresponding level and upsampling flow,
            # 		   upsampling flow]
            # then we obtain the final flow. 最终再加起来得到intLevel的flow
            tensorFlow = self.moduleBasic[intLevel](torch.cat([tensorFirst[intLevel],
                                                               Backward(tensorInput=tensorSecond[intLevel],
                                                                        tensorFlow=tensorUpsampled,
                                                                        cuda_flag=self.cuda_flag),
                                                               tensorUpsampled], 1)) + tensorUpsampled
        return tensorFlow


class warp(torch.nn.Module):
    def __init__(self, h, w, cuda_flag):
        super(warp, self).__init__()
        self.height = h
        self.width = w
        self.cuda_flag = cuda_flag
        # if cuda_flag:
        #     self.addterm = self.init_addterm().cuda()
        # else:
        #     self.addterm = self.init_addterm()

    def init_addterm(self):
        n = torch.FloatTensor(list(range(self.width)))
        horizontal_term = n.expand((1, 1, self.height, self.width))  # 第一个1是batch size
        n = torch.FloatTensor(list(range(self.height)))
        vertical_term = n.expand((1, 1, self.width, self.height)).permute(0, 1, 3, 2)
        addterm = torch.cat((horizontal_term, vertical_term), dim=1)
        return addterm

    def forward(self, frame, flow):
        """
        :param frame: frame.shape (batch_size=1, n_channels=3, width=256, height=448)
        :param flow: flow.shape (batch_size=1, n_channels=2, width=256, height=448)
        :return: reference_frame: warped frame
        """
        #####################################################
        self.height = frame.size()[2]
        self.width = frame.size()[3]
        if self.cuda_flag:
            self.addterm = self.init_addterm().cuda()
        else:
            self.addterm = self.init_addterm()
        #####################################################
        if True:
            flow = flow + self.addterm
        else:
            self.addterm = self.init_addterm()
            flow = flow + self.addterm

        horizontal_flow = flow[0, 0, :, :].expand(1, 1, self.height, self.width)  # 第一个0是batch size
        vertical_flow = flow[0, 1, :, :].expand(1, 1, self.height, self.width)

        horizontal_flow = horizontal_flow * 2 / (self.width - 1) - 1
        vertical_flow = vertical_flow * 2 / (self.height - 1) - 1
        flow = torch.cat((horizontal_flow, vertical_flow), dim=1)
        flow = flow.permute(0, 2, 3, 1)
        reference_frame = torch.nn.functional.grid_sample(frame, flow)
        return reference_frame

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=9, gc=32, bias=True):
      super(ResidualDenseBlock, self).__init__()
      # gc: growth channel, i.e. intermediate channels
      self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
      self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
      self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
      self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
      self.lrelu = nn.PReLU()

    def forward(self, x):
      x1 = self.lrelu(self.conv1(x))
      x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
      x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
      x4 = (self.conv4(torch.cat((x, x1, x2, x3), 1)))
      return x4*0.25+x1

class ResidualModule(nn.Module):
  def __init__(self,in_channels,kernel_size,dialtions=1, bias=False):
    super(ResidualModule, self).__init__()
    self.op = nn.Sequential(
      BasicConv(in_channels,in_channels,kernel_size, dilation=dialtions, relu=False,groups=in_channels),
      nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_channels,
                bias=False),
      nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(in_channels),
      nn.PReLU(),
      )
  def forward(self,x):
    res = self.op(x)
    return x+res

class EnhanceResidualModule(nn.Module):
  def __init__(self,in_channels,bias=False):
    super(EnhanceResidualModule, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=4, dilation=2, groups=in_channels, bias=False),
      nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_channels,
                bias=False),
      nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_channels,
                bias=False),
      nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(in_channels),
      nn.PReLU(),
      )
  def forward(self,x):
    res = self.op(x)
    return x+res
class NNalignedModule(torch.nn.Module):
  def __init__(self,c,cin=9):
    super(NNalignedModule, self).__init__()
    # Resblock groups
    self.preconv = torch.nn.Conv2d(in_channels=cin, out_channels=c, kernel_size=3, stride=1, padding=1)
    self.preconv2 = torch.nn.Conv2d(in_channels=cin, out_channels=c, kernel_size=3, stride=1, padding=1)
    # the blocks
    self.residual_1 = ResidualModule(c,3)
    self.residual_2 = EnhanceResidualModule(c)
    self.residual_3 = ResidualModule(c, 3)
    self.residual_4 = EnhanceResidualModule(c)
    self.conv_3x3_1 = torch.nn.Conv2d(in_channels=c*2, out_channels=c, kernel_size=3, padding=1)
    self.residual_5 = ResidualModule(c, 3)
    self.residual_6 = EnhanceResidualModule(c)
    # self.conv_3x3_2 = torch.nn.Conv2d(in_channels=c, out_channels=c//2, kernel_size=3, padding=1)
    # self.conv_1x1 = torch.nn.Conv2d(in_channels=c//2, out_channels=3, kernel_size=1)

  def forward(self,x):
    # Aligned_images
    x1 = torch.cat([x[:,0,:,:,:],x[:,2,:,:,:],x[:,4,:,:,:]],dim=1)
    x2 = torch.cat([x[:, 1, :, :, :], x[:, 2, :, :, :], x[:, 3, :, :, :]], dim=1)
    res1 =  self.preconv(x1)
    res2 =   self.preconv2(x2)
    res1 = self.residual_1(res1)
    res1 = self.residual_2(res1)
    res2 = self.residual_3(res2)
    res2 = self.residual_4(res2)
    res = torch.cat([res1,res2],dim=1)
    res = self.conv_3x3_1(res)
    res = self.residual_5(res)
    res = self.residual_6(res)
    # res = self.conv_3x3_2(res)
    # res = self.conv_1x1(res)
    return res
# the filter

class Cell_Decom_3(nn.Module):

  def __init__(self, steps, cin, C):
    super(Cell_Decom_3, self).__init__()
    self._C = C
    self._steps = steps  # inner nodes
    self.radiux = [4,6]
    self.eps_list = [0.001, 0.0001]
    self.res_block1 = EnhanceResidualModule(C)
    self.res_block2 = ECABasicBlock(C,C)
    self.res_block3 = EnhanceResidualModule(C)
    self.res_block4 = EnhanceResidualModule(C)
    self.res_block5 = ECABasicBlock(C,C)
    self.res_block6 = EnhanceResidualModule(C)
    self.preconv = torch.nn.Conv2d(in_channels=cin, out_channels=C, kernel_size=3, stride=1, padding=1)
    self.conv1x1_lf = nn.Conv2d(C * 4, C, kernel_size=1, bias=False)
    self.conv1x1_hf = nn.Conv2d(C * 4, C, kernel_size=1, bias=False)
    self.conv1x1_concat = nn.Conv2d(C * 2, C, kernel_size=1, bias=False)
    self.enchance_concat = EnhanceResidualModule(C)

  def forward(self, inp_features):

    inp_reduce = self.preconv(inp_features)

    lf, hf = self.decomposition(inp_reduce, self._C)
    lf = self.conv1x1_lf(lf)
    hf = self.conv1x1_hf(hf)

    sl1 = self.res_block1(lf)
    sl2 = self.res_block2(sl1)
    lf = self.res_block3(sl2)
    sl3 = self.res_block4(hf)
    sl4 = self.res_block5(sl3)
    hf = self.res_block6(sl4)
    fea_cat = torch.cat([lf, hf], dim=1)
    feature = self.conv1x1_concat(fea_cat)
    feature_res = self.enchance_concat(feature)
    return inp_reduce + feature_res

  def get_residue(self, tensor):
    max_channel = torch.max(tensor, dim=1, keepdim=True)
    min_channel = torch.min(tensor, dim=1, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel

  def decomposition(self, x, C):
    LF_list = []
    HF_list = []
    res = self.get_residue(x)
    res = res.repeat(1, C, 1, 1)
    for radius in self.radiux:
      for eps in self.eps_list:
        self.gf = GuidedFilter(radius, eps)
        LF = self.gf(res, x)
        LF_list.append(LF)
        HF_list.append(x - LF)
    LF = torch.cat(LF_list, dim=1)
    HF = torch.cat(HF_list, dim=1)
    return LF, HF

class MixedOp(nn.Module):

  def __init__(self, C, primitive):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    kernel = 3
    dilation = 1
    if primitive.find('attention') != -1:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
    else:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
        dilation = int(primitive.split('_')[2])
    print(name, kernel, dilation)
    self._op = OPS[name](C, kernel, dilation, False)

  def forward(self, x):
    return self._op(x)

class Cell_Fusion(nn.Module):

  def __init__(self, C,type,concat):
    super(Cell_Fusion, self).__init__()
    op_names, indices = zip(*type)
    concat = concat
    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names)
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      print(name,index)
      stride = 1
      op = MixedOp(C,name)
      self._ops += [op]
    self._indices = indices

  def forward(self, inp_features):
    offset = 0
    s1 = inp_features
    for i in range(self._steps):
      s1 = self._ops[offset](s1)
      offset += 1
    return inp_features+s1


class ResBlock(torch.nn.Module):
  def __init__(self,Cin=15,C_out=128):
      super(ResBlock, self).__init__()
      self.conv1 = torch.nn.Conv2d(in_channels=Cin, out_channels=C_out, kernel_size=3, stride=1, padding=1)
      self.conv2 = torch.nn.Conv2d(in_channels=C_out, out_channels=C_out, kernel_size=3, stride=1, padding=1)
      self.relu = nn.PReLU()
  def forward(self, frames):
      res = self.conv1(frames)
      res = self.relu(res)
      res = self.conv2(res)
      return  res
import scipy.io as io
kernel = io.loadmat('./init_kernel.mat')['C9']  # 3*32*9*9
kernel = torch.FloatTensor(kernel)

# filtering on rainy image for initializing B^(0) and Z^(0), refer to supplementary material(SM)
w_x = (torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9)
w_x_conv = w_x.unsqueeze(dim=0).unsqueeze(dim=0)

class TOFlow(torch.nn.Module):
    def __init__(self, h, w, args, task, cuda_flag, genotype):
        super(TOFlow, self).__init__()
        self.height = h
        self.width = w
        self.task = task
        self.cuda_flag = cuda_flag

        # if config.train_or_eval is False:
        #     self.is_mask_filter = True     # True
        # else:
        #     self.is_mask_filter = False

        # self.raft = RAFT(args)
        # self.raft.load_state_dict(torch.load(args.model))
        # # # self.raft = self.raft.module
        # # # self.raft.load_state_dict(torch.load(args.model, ))  # map_location='cpu'
        # # # self.SpyNet = SpyNet(cuda_flag=self.cuda_flag)  # SpyNet层
        # # for param in self.raft.parameters():  # fix
        # #     param.requires_grad = False
        # #
        # self.warp = warp(self.height, self.width, cuda_flag=self.cuda_flag)
        # self.feature = Feature_CARB()
        # self.ResNet = ResNet(task=self.task)
        # self.cleaner = cleaner()
        # self.flow_pwc = Flow_PWC()
        # self.unet = U_Net(in_ch=config.N * 3 + 0, out_ch=3)     #
        # self.fjdn = FJDN()
        # self.feature = Feature_CARB_mutil(in_ch=9, out_ch=3)
        C = 96
        self.feature = NNalignedModule(c=96)
        self.cell_1 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        self.cell_2 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        self.cell_3 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        self.cell_4 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        self.cell_5 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        self.cell_6 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        self.cell_7 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        self.cell_8 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        self.decom = Cell_Decom_3(4, C, C)
        self.decom2 = Cell_Decom_3(4, C, C)
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=True)
        )
        self.stem_out = nn.Sequential(
            nn.Conv2d(C, C//2, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(C//2, 32, 3, padding=1, bias=True),
        )
        self.stem_out2 = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(C // 2, 3, 3, padding=1, bias=True),
        )
        self.stem_out3 = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(C // 2, 3, 3, padding=1, bias=True),
        )
        self.tanh = nn.Tanh()
        # Rain kernel
        self.weight0 = nn.Parameter(data=kernel, requires_grad=True)  # used in initialization process
        self.conv = self.make_weight(2,
                                     kernel)  # rain kernel is inter-stage sharing. The true net parameter number is (#self.conv /self.iter)

    # frames should be TensorFloat
    # frames should be TensorFloat
    def make_weight(self, iters,const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters,-1,-1,-1,-1)
        weight = nn.Parameter(data=const_f, requires_grad = True)
        return weight
    def forward(self, frames, iters=12, test_mode=False, opticalflow = False):
        """
        :param frames: [batch_size=1, img_num=3, n_channels=3, h, w]
        :return: img_tensor:
        """
        # print('**', frames.shape)
        # if opticalflow:
        #     for i in range(frames.size(1)):
        #         frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])
        #     if self.cuda_flag:
        #         opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4)).cuda()
        #         warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4)).cuda()
        #     else:
        #         opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4))
        #         warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4))
        #     if test_mode is False:
        #         process_index = []
        #         for i in range(config.N):
        #             if i != (config.N // 2):
        #                 process_index.append(i)
        #         for i in process_index:
        #             opticalflows[:, i, :, :, :] = self.raft(frames[:, config.N // 2, :, :, :], frames[:, i, :, :, :])[11]
        #         warpframes[:, config.N // 2, :, :, :] = frames[:, config.N // 2, :, :, :]
        #         for i in process_index:
        #             warpframes[:, i, :, :, :] = self.warp(frames[:, i, :, :, :], opticalflows[:, i, :, :, :])
        #     else:
        #         process_index = []
        #         for i in range(config.N):
        #             if i != (config.N // 2):
        #                 process_index.append(i)
        #         for i in process_index:
        #             _, opticalflows[:, i, :, :, :] = self.raft(frames[:, config.N // 2, :, :, :], frames[:, i, :, :, :], iters=iters, test_mode=test_mode)
        #         warpframes[:, config.N // 2, :, :, :] = frames[:, config.N // 2, :, :, :]
        #         for i in process_index:
        #             warpframes[:, i, :, :, :] = self.warp(frames[:, i, :, :, :], opticalflows[:, i, :, :, :])
        #
        #         # warped01, _, _, flow_mask01 = self.flow_pwc(frames[:, 1, :, :, :], frames[:, 0, :, :, :])
        #         # warped21, _, _, flow_mask21 = self.flow_pwc(frames[:, 1, :, :, :], frames[:, 2, :, :, :])
        #         #
        #         # one_mask = torch.ones_like(flow_mask01)
        #         # frame_warp_list = [warped01, frames[:, 1, :, :, :], warped21]
        #         # flow_mask_list = [flow_mask01, one_mask.detach(), flow_mask21]
        #         # luckiness = self.get_masks(frame_warp_list, flow_mask_list)     # Temporal Sharpness Prior
        #         # # print('luckness shape: ', luckiness.shape, luckiness.dtype)     # channel=1
        #         # concated = torch.cat([warped01, frames[:, 1, :, :, :], warped21, luckiness], dim=1)
        #
        #         # feature = self.feature(frames[:, config.N // 2, :, :, :])
        #         # x = feature
        #         # for i in range(0, warpframes.size(1)):
        #         #     x = torch.cat((x, warpframes[:, i, :, :, :]), dim=1)
        #         # warpframes = x  # warpframes: 1x18xHxW
        #         x = torch.cat((warpframes[:, 0, :, :, :], warpframes[:, 1, :, :, :], warpframes[:, 2, :, :, :]), dim=1)
        #         # x = torch.cat((frames[:, 0, :, :, :], frames[:, 1, :, :, :], frames[:, 2, :, :, :]), dim=1)
        # else:
        x = frames[:, 2, :, :, :]
        res = self.feature(frames)
        res = self.decom(res)
        res_1 = self.stem_out(res)

        rain_streak = F.conv2d(res_1, self.conv[1, :, :, :, :] / 10, stride=1,
                                   padding=4)  # self.conv[1,:,:,:,:]：rain kernel is inter-stage sharing
        # residule and orignal
        # x_rob = x - rain_streak
        # x_concat = torch.cat([x, rain_streak], dim=1)
        s1 = self.stem(rain_streak) +res
        s1 = self.cell_1(s1)
        s1 = self.cell_2(s1)
        s1 = self.cell_3(s1)
        s1 = self.cell_4(s1)
        # s1 = self.cell_5(s1)
        s1 = self.tanh(self.stem_out2(s1))
        res1 = x + s1

        s2 = self.cell_5(res)
        s2 = self.cell_6(s2)
        s2 = self.cell_7(s2)
        s2 = self.cell_8(s2)
        s2 = self.tanh(self.stem_out3(s2))
        res2 = x + s2
        return res1, res2
        # 现在 Img: batch_size=1, c=3, H, W



class TOFlow3(torch.nn.Module):
    def __init__(self, h, w, args, task, cuda_flag, genotype):
        super(TOFlow3, self).__init__()
        self.height = h
        self.width = w
        self.task = task
        self.cuda_flag = cuda_flag

        # if config.train_or_eval is False:
        #     self.is_mask_filter = True     # True
        # else:
        #     self.is_mask_filter = False

        # self.raft = RAFT(args)
        # self.raft.load_state_dict(torch.load(args.model))
        # # # self.raft = self.raft.module
        # # # self.raft.load_state_dict(torch.load(args.model, ))  # map_location='cpu'
        # # # self.SpyNet = SpyNet(cuda_flag=self.cuda_flag)  # SpyNet层
        # # for param in self.raft.parameters():  # fix
        # #     param.requires_grad = False
        # #
        # self.warp = warp(self.height, self.width, cuda_flag=self.cuda_flag)
        # self.feature = Feature_CARB()
        # self.ResNet = ResNet(task=self.task)
        # self.cleaner = cleaner()
        # self.flow_pwc = Flow_PWC()
        # self.unet = U_Net(in_ch=config.N * 3 + 0, out_ch=3)     #
        # self.fjdn = FJDN()
        # self.feature = Feature_CARB_mutil(in_ch=9, out_ch=3)
        C = 96
        self.feature = NNalignedModule(c=96)
        self.cell_1 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        self.cell_2 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        # self.cell_3 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)
        # self.cell_4 = Cell_Fusion(C, genotype.normal_2, genotype.normal_2_concat)

        self.decom = Cell_Decom_3(4, C, C)
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=True)
        )
        self.stem_out = nn.Sequential(
            nn.Conv2d(C, C//2, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(C//2, 32, 3, padding=1, bias=True),
        )
        self.stem_out2 = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(C // 2, 3, 3, padding=1, bias=True),
        )
        self.tanh = nn.Tanh()
        # Rain kernel
        self.weight0 = nn.Parameter(data=kernel, requires_grad=True)  # used in initialization process
        self.conv = self.make_weight(2,
                                     kernel)  # rain kernel is inter-stage sharing. The true net parameter number is (#self.conv /self.iter)

    # frames should be TensorFloat
    # frames should be TensorFloat
    def make_weight(self, iters,const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters,-1,-1,-1,-1)
        weight = nn.Parameter(data=const_f, requires_grad = True)
        return weight
    def forward(self, frames, iters=12, test_mode=False, opticalflow = False):
        """
        :param frames: [batch_size=1, img_num=3, n_channels=3, h, w]
        :return: img_tensor:
        """
        # print('**', frames.shape)
        # if opticalflow:
        #     for i in range(frames.size(1)):
        #         frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])
        #     if self.cuda_flag:
        #         opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4)).cuda()
        #         warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4)).cuda()
        #     else:
        #         opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4))
        #         warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4))
        #     if test_mode is False:
        #         process_index = []
        #         for i in range(config.N):
        #             if i != (config.N // 2):
        #                 process_index.append(i)
        #         for i in process_index:
        #             opticalflows[:, i, :, :, :] = self.raft(frames[:, config.N // 2, :, :, :], frames[:, i, :, :, :])[11]
        #         warpframes[:, config.N // 2, :, :, :] = frames[:, config.N // 2, :, :, :]
        #         for i in process_index:
        #             warpframes[:, i, :, :, :] = self.warp(frames[:, i, :, :, :], opticalflows[:, i, :, :, :])
        #     else:
        #         process_index = []
        #         for i in range(config.N):
        #             if i != (config.N // 2):
        #                 process_index.append(i)
        #         for i in process_index:
        #             _, opticalflows[:, i, :, :, :] = self.raft(frames[:, config.N // 2, :, :, :], frames[:, i, :, :, :], iters=iters, test_mode=test_mode)
        #         warpframes[:, config.N // 2, :, :, :] = frames[:, config.N // 2, :, :, :]
        #         for i in process_index:
        #             warpframes[:, i, :, :, :] = self.warp(frames[:, i, :, :, :], opticalflows[:, i, :, :, :])
        #
        #         # warped01, _, _, flow_mask01 = self.flow_pwc(frames[:, 1, :, :, :], frames[:, 0, :, :, :])
        #         # warped21, _, _, flow_mask21 = self.flow_pwc(frames[:, 1, :, :, :], frames[:, 2, :, :, :])
        #         #
        #         # one_mask = torch.ones_like(flow_mask01)
        #         # frame_warp_list = [warped01, frames[:, 1, :, :, :], warped21]
        #         # flow_mask_list = [flow_mask01, one_mask.detach(), flow_mask21]
        #         # luckiness = self.get_masks(frame_warp_list, flow_mask_list)     # Temporal Sharpness Prior
        #         # # print('luckness shape: ', luckiness.shape, luckiness.dtype)     # channel=1
        #         # concated = torch.cat([warped01, frames[:, 1, :, :, :], warped21, luckiness], dim=1)
        #
        #         # feature = self.feature(frames[:, config.N // 2, :, :, :])
        #         # x = feature
        #         # for i in range(0, warpframes.size(1)):
        #         #     x = torch.cat((x, warpframes[:, i, :, :, :]), dim=1)
        #         # warpframes = x  # warpframes: 1x18xHxW
        #         x = torch.cat((warpframes[:, 0, :, :, :], warpframes[:, 1, :, :, :], warpframes[:, 2, :, :, :]), dim=1)
        #         # x = torch.cat((frames[:, 0, :, :, :], frames[:, 1, :, :, :], frames[:, 2, :, :, :]), dim=1)
        # else:
        x = frames[:, 2, :, :, :]
        res = self.feature(frames)
        res = self.decom(res)
        res_1 = self.stem_out(res)

        rain_streak = F.conv2d(res_1, self.conv[1, :, :, :, :] / 10, stride=1,
                                   padding=4)  # self.conv[1,:,:,:,:]：rain kernel is inter-stage sharing
        # residule and orignal
        # x_rob = x - rain_streak
        # x_concat = torch.cat([x, rain_streak], dim=1)
        s1 = self.stem(rain_streak) +res
        s1 = self.cell_1(s1)
        s1 = self.cell_2(s1)
        # s1 = self.cell_3(s1)
        # s1 = self.cell_4(s1)
        # s1 = self.cell_5(s1)
        s1 = self.tanh(self.stem_out2(s1))
        res1 = x + s1
        return res1, res1
        # 现在 Img: batch_size=1, c=3, H, W

class SoftAttention(nn.Module):
  def __init__(self, channel, stride=1, affine=False, reduction=64):
    super(SoftAttention, self).__init__()
    self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, reduction),
      nn.ReLU(inplace=True),
      nn.Linear(reduction, channel),
      nn.Sigmoid()
    )
    self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, stride=stride, padding=0, bias=False,)
    self.bn = nn.BatchNorm2d(channel, affine=affine)
  def forward(self, image1, image2):
    # tvs = get_tv(x)
    b, c, _, _ = image1.size()
    alpha =  self.conv1(image1)
    alpha2 = self.conv1(image2)
    y = self.avg_pool_1(alpha).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    y2 = self.avg_pool_1(alpha2).view(b, c)
    y2 = self.fc(y2).view(b, c, 1, 1)
    # y =x*y
    return y,y2

class TOFlow2(torch.nn.Module):

    def __init__(self, h, w, args, task, cuda_flag, genotype):
        super(TOFlow2, self).__init__()
        self.height = h
        self.width = w
        self.task = task
        self.cuda_flag = cuda_flag
        self.toflow = TOFlow(h, w, args, task, cuda_flag, genotype)
        # self.toflow.load_state_dict(torch.load(model_path)['net_state_dict'])
        for param in self.toflow.parameters():  # fix
            param.requires_grad = False
        self.attention = SoftAttention(3)

    def forward(self, frames, iters=12, test_mode=False, opticalflow=True):
        """
        :param frames: [batch_size=1, img_num=3, n_channels=3, h, w]
        :return: img_tensor:
        """
        # print('**', frames.shape)
        img1, img2 = self.toflow(frames, iters, test_mode, opticalflow)
        alpha1, alpha2 = self.attention(img1, img2)
        # setting the weights
        alpha1 = alpha1 * 1e-1
        # print(alpha1)
        # alpha1 = 0.7
        return  alpha1*img1 + (1 - alpha1)  * img2
    # def warp(self, x, flo):
    #     """
    #     warp an image/tensor (im2) back to im1, according to the optical flow
    #         x: [B, C, H, W] (im2)
    #         flo: [B, 2, H, W] flow
    #     """
    #     B, C, H, W = x.size()
    #     # mesh grid
    #     xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    #     yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    #     xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    #     yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    #     grid = torch.cat((xx, yy), 1).float().cuda()
    #     # grid = grid.to(self.device)
    #     vgrid = Variable(grid) + flo
    #
    #     # scale grid to [-1,1]
    #     vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    #     vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    #
    #     vgrid = vgrid.permute(0, 2, 3, 1)
    #     output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
    #     mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    #     mask = nn.functional.grid_sample(mask, vgrid)
    #
    #     mask[mask < 0.999] = 0
    #     mask[mask > 0] = 1
    #
    #     output = output * mask
    #
    #     return output, mask
    #
    # def get_masks(self, img_list, flow_mask_list):
    #     """
    #     得到 Temporal Sharpness Prior
    #     Args:
    #         img_list:
    #         flow_mask_list:
    #
    #     Returns:
    #
    #     """
    #     num_frames = len(img_list)
    #
    #     img_list_copy = [img.detach() for img in img_list]  # detach backward
    #     if self.is_mask_filter:  # mean filter
    #         img_list_copy = [utils.calc_meanFilter(im, n_channel=3, kernel_size=5) for im in img_list_copy]
    #
    #     delta = 1.
    #     mid_frame = img_list_copy[num_frames // 2]
    #     diff = torch.zeros_like(mid_frame).cuda()
    #     for i in range(num_frames):
    #         diff = diff + (img_list_copy[i] - mid_frame).pow(2)
    #     diff = diff / (2 * delta * delta)
    #     diff = torch.sqrt(torch.sum(diff, dim=1, keepdim=True))
    #     luckiness = torch.exp(-diff)  # (0,1)
    #
    #     sum_mask = torch.ones_like(flow_mask_list[0]).cuda()
    #     for i in range(num_frames):
    #         sum_mask = sum_mask * flow_mask_list[i]
    #     sum_mask = torch.sum(sum_mask, dim=1, keepdim=True)
    #     sum_mask = (sum_mask > 0).float()
    #     luckiness = luckiness * sum_mask
    #
    #     return luckiness
