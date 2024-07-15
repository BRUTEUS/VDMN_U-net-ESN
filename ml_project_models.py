import time
import argparse
from time import sleep
import os
# from __future__ import print_function, division
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from tqdm import tqdm
import math

# nadam not available in pytorch lib, SO pytroch implementation below:
import math
import torch
# from optimizer import Optimizer
from torch.optim import Optimizer

# load:
import numpy as np
# from sklearn.preprocessing import OrdinalEncoder
# # USING DNS FOR EMBEDDING:
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.feature_extraction import FeatureHasher
from tqdm import tqdm
from sklearn.externals import joblib
import ujson
import json
import pickle
import csv
from multiprocessing import Pool, freeze_support

# ------------------------------------
import pandas as pd
import random
import itertools
from collections import Counter

from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import torch
from torchviz import make_dot
import torchvision
import torch.autograd as autograd
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
from sklearn.utils import shuffle
import math
from torch.autograd import Variable
import torchvision.models as models
# from optimizer import Optimizer
from torch.optim import Optimizer
import hiddenlayer as hlayer
from qhoptim.pyt import QHM, QHAdam
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import RMSprop
from torch.optim import Adagrad
from torch.optim import Adadelta
from torchvision.utils import save_image
import torch
import torchvision
from math import log10

rand_tensor= torch.rand(64, 3,28,28)

from torchvision.utils import save_image



# Parsing helpers
import string
# import nltk
import re
import sys

# device = torch.device("cuda")
device = torch.device("cpu")


from torch.autograd import Variable


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

    # if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
    #     for name, param in m.named_parameters():
    #         if 'weight_ih' in name:
    #             torch.nn.init.xavier_uniform_(param.data)
    #         elif 'weight_hh' in name:
    #             torch.nn.init.orthogonal_(param.data)
    #         elif 'bias' in name:
    #             param.data.fill_(0)

    ## if isinstance(m, nn.GRU):
    ##     torch.nn.init.xavier_uniform_(m.weight)
    ##     torch.nn.init.zeros_(m.bias)


class Nadam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):

        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Nadam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                # State initialization

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prod_mu_t'] = 1.

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # prod_mu_t = 1
                mu_t = beta1 * (1 - 0.5 * 0.96 ** (state['step'] / 250))
                mu_t_1 = beta1 * (1 - 0.5 * 0.96 ** ((state['step'] + 1) / 250))
                prod_mu_t = state['prod_mu_t'] * mu_t
                prod_mu_t_1 = prod_mu_t * mu_t_1
                state['prod_mu_t'] = prod_mu_t
                # for i in range(state['step']):
                #     mu_t = beta1*(1 - 0.5*0.96**(i/250))
                #     mu_t_1 = beta1*(1 - 0.5*0.96**((i+1)/250))
                #     prod_mu_t = prod_mu_t * mu_t
                #     prod_mu_t_1 = prod_mu_t * mu_t_1
                g_hat = grad / (1 - prod_mu_t)
                m_hat = exp_avg / (1 - prod_mu_t_1)
                m_bar = (1 - mu_t) * g_hat + mu_t_1 * m_hat
                exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** state['step'])
                denom = exp_avg_sq_hat.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                step_size = group['lr']
                p.data.addcdiv_(-step_size, m_bar, denom)
                # p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss


class SRResNet(nn.Module):
    def __init__(self, num_channels, num_resid_layers):
        super(SRResNet, self).__init__()


    #     -------------------------------------------------------------
    #     -------------------------------------------------------------
    #     -------------------------------------------------------------
    # SRCNN FOR LR TO HR MAPPING
        # N x Cin x D x H x W
        # self.num_channels_lr_hr_mapping = 64
        self.num_channels_lr_hr_mapping = num_channels
        # self.num_resid_layers = 2
        # self.num_resid_layers = 16
        self.num_resid_layers = num_resid_layers

        self.conv_input_lr_hr_map = nn.Conv2d(in_channels=3, out_channels=self.num_channels_lr_hr_mapping, kernel_size=9, stride=1,
                                    padding=4, bias=False)
        self.relu_lr_hr_map = nn.LeakyReLU(0.2, inplace=True)

        # self.residual = self.make_layer(_Residual_Block, 16)
        # self.residual = self.make_layer(_Residual_Block, 2)
        self.residual_lr_hr_map = self.make_layer(_Residual_Block_Efficient_2d, self.num_channels_lr_hr_mapping, num_resid_layers)

        # self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_mid_lr_hr_map = nn.Conv2d(in_channels=self.num_channels_lr_hr_mapping,
                                  out_channels=self.num_channels_lr_hr_mapping, kernel_size=3, stride=1, padding=1,
                                  bias=False)
        # self.bn_mid = nn.InstanceNorm2d(64, affine=True)
        self.bn_mid_lr_hr_map = nn.InstanceNorm2d(self.num_channels_lr_hr_mapping, affine=True)

        self.upscale4x_lr_hr_map = nn.Sequential(
            # N x Cin x H x W
            # nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=self.num_channels_lr_hr_mapping, out_channels=256, kernel_size=3, stride=1, padding=1,
                      bias=False),
            # nn.Conv3d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.num_channels_lr_hr_mapping, out_channels=256, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output_lr_hr_map = nn.Conv2d(in_channels=self.num_channels_lr_hr_mapping, out_channels=3, kernel_size=9, stride=1,
                                     padding=4, bias=False)



    #     -------------------------------------------------------------
    #     -------------------------------------------------------------
    #     -------------------------------------------------------------

    def model_params(self, debug=True):
        print('model parameters: ', end='')
        params = []
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for p in p_list:
                out *= p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())
            if debug:
                print(p.requires_grad, p.size())
        print('%s\n' % '{:,}'.format(total_size))
        return params


    def make_layer(self, block, input_channel_size, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(input_channel_size))
        return nn.Sequential(*layers)


    def forward(self, x, mask):

        print('WITHIN SRRESNET')
        print(x.shape)
        print(mask.shape)
        # applied_mask = mask * x
        print(x)
        # print(mask)
        # applied_mask = mask * x
        # applied_mask = mask + x
        applied_mask = x
        out = self.relu_lr_hr_map(self.conv_input_lr_hr_map(applied_mask))
        print('WITHIN SRRESNET BEFORE RESIDUAL')
        print(out.shape)
        # residual = out

        # out = out * mask
        # out = out + (mask*0.01)
        out = out + (out*mask*0.01)
        residual = out

        # print(out.shape)

        out = self.residual_lr_hr_map(out)
        out = self.bn_mid_lr_hr_map(self.conv_mid_lr_hr_map(out))
        out = torch.add(out, residual)
        print('WITHIN SRRESNET BEFORE UPSCALE')
        print(out.shape)
        # out = out * mask

        out = self.upscale4x_lr_hr_map(out)


        out = self.conv_output_lr_hr_map(out)

        return out




class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.in1 = nn.InstanceNorm3d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.in2 = nn.InstanceNorm3d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class _Residual_Block_Efficient(nn.Module):
    def __init__(self, num_in_channels):
        super(_Residual_Block_Efficient, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv3d(in_channels=num_in_channels, out_channels=num_in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.in1 = nn.InstanceNorm3d(num_in_channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(in_channels=num_in_channels, out_channels=num_in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.in2 = nn.InstanceNorm3d(num_in_channels, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output

class _Residual_Block_Efficient_2d(nn.Module):
    def __init__(self, num_in_channels):
        super(_Residual_Block_Efficient_2d, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=num_in_channels, out_channels=num_in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.in1 = nn.InstanceNorm2d(num_in_channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_in_channels, out_channels=num_in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.in2 = nn.InstanceNorm2d(num_in_channels, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output

# embed_size, decoder_input_size_sequence_length,\
#     decoder_input_sequence_elem_length
def get_encoder_and_decoder_input_sizes(train_x_np, encoder_cnn_feat_model, window_size):

    # for batch_idx, ((data_x, false_folder_labels_x), (data_y, false_folder_labels_y)) in enumerate(zip(train_loader_x, train_loader_y)):
    # for batch_idx, (data_x, false_folder_labels_x) in train_x_np:
    for batch_idx, ((data_x, false_folder_labels_x), (data_y, false_folder_labels_y)) in enumerate(zip(train_x_np, train_x_np)):
        print('data:')
        data_shape = data_x.shape
        print(data_shape)
        curr_target_image_pointer = window_size - 1
        curr_window_x, curr_window_y = get_current_window_encoder_decoder(data_x, data_x, window_size,
                                                                          curr_target_image_pointer)
        # y_true_i = curr_window_y
        # s_lens = curr_window_x.shape[0]
        # q_lens = curr_window_y.shape[0]
        # e_lens = window_size

        curr_window_x = torch.tensor(curr_window_x, dtype=torch.float)
        # N x Cin x D x H x W
        curr_window_x = curr_window_x.view(1, curr_window_x.shape[1], curr_window_x.shape[0],
                                           curr_window_x.shape[2], curr_window_x.shape[3])
        # curr_window_y = curr_window_y.view(1, curr_window_y.shape[1], curr_window_y.shape[0],
        #                                    curr_window_y.shape[2], curr_window_y.shape[3])

        output = encoder_cnn_feat_model(curr_window_x)
        output_shape = output.shape
        break
    encoder_input_embed_size = output_shape[3]
    decoder_input_size_sequence_length = output_shape[1]
    decoder_input_sequence_elem_length = output_shape[3]
    return encoder_input_embed_size, decoder_input_size_sequence_length, decoder_input_sequence_elem_length



    # # encoder_input_size = decoder_input_size = train_x_np.shape[1]
    # encoder_input_size = train_x_np.shape[3]
    # decoder_input_size = train_x_np.shape[1]
    # decoder_input_sequence_length = train_x_np.shape[3]
    # return encoder_input_size, decoder_input_size, decoder_input_sequence_length

# for step, (torch_window_arr_x, torch_window_arr_y) in enumerate(train_loader):
def get_encoder_and_decoder_input_sizes_loader(train_x_np, encoder_cnn_feat_model, window_size):
    # for batch_idx, ((data_x, false_folder_labels_x), (data_y, false_folder_labels_y)) in enumerate(zip(train_loader_x, train_loader_y)):
    for batch_idx, (data_x, y) in enumerate(train_x_np):
    # for batch_idx, ((data_x, false_folder_labels_x), (data_y, false_folder_labels_y)) in enumerate( zip(train_x_np, train_x_np)):
        print('data:')
        data_shape = data_x.shape
        print(data_shape)
        curr_target_image_pointer = window_size - 1
        curr_window_x, curr_window_y = get_current_window_encoder_decoder(data_x, data_x, window_size,
                                                                          curr_target_image_pointer)
        # y_true_i = curr_window_y
        # s_lens = curr_window_x.shape[0]
        # q_lens = curr_window_y.shape[0]
        # e_lens = window_size

        # curr_window_x = torch.tensor(curr_window_x, dtype=torch.float)
        # # N x Cin x D x H x W
        # curr_window_x = curr_window_x.view(1, curr_window_x.shape[1], curr_window_x.shape[0],
        #                                    curr_window_x.shape[2], curr_window_x.shape[3])
        # # curr_window_y = curr_window_y.view(1, curr_window_y.shape[1], curr_window_y.shape[0],
        # #                                    curr_window_y.shape[2], curr_window_y.shape[3])

        output = encoder_cnn_feat_model(curr_window_x)
        output_shape = output.shape
        break
    encoder_input_embed_size = output_shape[3]
    decoder_input_size_sequence_length = output_shape[1]
    decoder_input_sequence_elem_length = output_shape[3]
    return encoder_input_embed_size, decoder_input_size_sequence_length, decoder_input_sequence_elem_length


def train_full_batch_reference(model, model_optimizer, criterion, train_loader_x, train_loader_y, iter):
    model.train()
    batch_num = 0
    every_N_print = 1
    total_loss = 0
    # window_size = 4
    window_size = model.max_slen
    window_size_q = model.max_qlen
    # num_clips_per_vid = train_loader_x.shape[0]

    for batch_idx, ((data_x, false_folder_labels_x), (data_y, false_folder_labels_y)) in enumerate(zip(train_loader_x, train_loader_y)):
        print('data:')
        print(data_x.shape)
        print(false_folder_labels_x)
        print('target')
        print(data_y.shape)
        print(false_folder_labels_y)
        num_clips_per_vid = data_x.shape[0]

        batch_start_time = time.time()
        if (batch_num % every_N_print == 0):
            print('BATCH NUM:')
            print(batch_num)
            print()
        batch_num += 1
        loss = 0
        model_optimizer.zero_grad()

        curr_target_image_pointer = window_size - 1
        y_hat_arr = []
        y_true_arr = []

        # curr_window_x_arr = []
        # curr_window_y_arr = []

        curr_sample_count_i = 0
        while(curr_target_image_pointer < num_clips_per_vid):
            curr_window_x, curr_window_y = get_current_window(data_x, data_y, window_size, window_size_q, curr_target_image_pointer)
            y_true_i = curr_window_y[-1, :, :, :]

            s_lens = curr_window_x.shape[0]
            q_lens = curr_window_y.shape[0]
            e_lens = window_size

            curr_window_x = torch.tensor(curr_window_x, dtype=torch.float)
            # N x Cin x D x H x W
            curr_window_x = curr_window_x.view(1, curr_window_x.shape[1], curr_window_x.shape[0], curr_window_x.shape[2], curr_window_x.shape[3])
            curr_window_y = curr_window_y.view(1, curr_window_y.shape[1], curr_window_y.shape[0], curr_window_y.shape[2], curr_window_y.shape[3])

            # y_hat_i, gates = model(curr_window_x, curr_window_y[:, :, :-1, :, :], s_lens, q_lens, e_lens)
            y_hat_i, gates = model(curr_window_x, curr_window_y, s_lens, q_lens, e_lens)
            print('computed model output')
            print('CURRENT SAMPLE NUM')
            print(curr_sample_count_i)
            del(curr_window_x)
            del(curr_window_y)



            y_true_arr.append(y_true_i)
            y_hat_arr.append(y_hat_i)
            curr_target_image_pointer += 1

        y_true_arr = np.vstack(y_true_arr)
        y_true_arr = torch.tensor(y_true_arr, dtype=torch.long)
        y_true_arr = y_true_arr.to(device)

        # y_hat_arr = np.vstack(y_hat_arr)
        y_hat_arr = torch.stack(y_hat_arr)
        y_hat_arr = torch.tensor(y_hat_arr, dtype=torch.long)
        y_hat_arr = y_hat_arr.to(device)

        loss = criterion(y_hat_arr, y_true_arr)
        # reconstr_loss = criterion(x_bar, x)
        # kl_loss = F.kl_div(q.log(), p[idx])

        total_loss += loss
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), model.config.grad_max_norm)
        model_optimizer.step()
        delta_t_minutes = (time.time() - batch_start_time) / 60.0
        print('-------------- FULL ELAPSED TIME FOR SINGLE BATCH = %f minutes ------------------' % delta_t_minutes)

        # save model:
        # print('SAVING MODEL AFTER SINGLE EPOCH TRAINING')
        # save_file_encoder = '/shared/encoder_model_save_file.pth'
        # save_file_decoder = '/shared/decoder_model_save_file.pth'

        loss_curr = loss.item()
        print('BATCH LOSS:')
        print(loss_curr)
        print('BATCH LOSS - AVERAGE LOSS PER SAMPLE:')
        loss_avg_samples_per_batch = loss_curr / (num_clips_per_vid - window_size + 1)
        print(loss_avg_samples_per_batch)
        print('EPOCH NUMBER:')
        print(iter)
        print()

        # torch.save(encoder, save_file_encoder)
        # torch.save(decoder, save_file_decoder)

    loss_avg_total = total_loss / batch_num
    print('AVG LOSS ACROSS EPOCH - AVG LOSS PER BATCH:')
    print(loss_avg_total)

    return model, loss_avg_total


def get_current_window_encoder_decoder(data_x, data_y, window_size, curr_target_image_pointer):
    start_ind = curr_target_image_pointer - window_size + 1
    end_ind = curr_target_image_pointer + 1
    window_x = data_x[start_ind:end_ind, :, :, :]
    # end_ind_y = end_ind - 1
    # # start_ind_y = curr_target_image_pointer - window_size_q + 1 - 1
    # start_ind_y = curr_target_image_pointer - window_size_q
    # # start_ind_y = end_ind - 2
    window_y_target = data_y[-1, :, :, :]
    return window_x, window_y_target



def create_X_and_Y(train_loader_x, train_loader_y, window_size):

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    curr_x_list = []
    curr_y_list = []
    batch_num = 0
    every_N_print = 1
    for batch_idx, ((data_x, false_folder_labels_x), (data_y, false_folder_labels_y)) in enumerate(zip(train_loader_x, train_loader_y)):
        print('data:')
        print(data_x.shape)
        print(false_folder_labels_x)
        print('target')
        print(data_y.shape)
        print(false_folder_labels_y)
        num_clips_per_vid = data_x.shape[0]

        batch_start_time = time.time()
        if (batch_num % every_N_print == 0):
            print('BATCH NUM DATA PROCESSING:')
            print(batch_num)
            print()
        batch_num += 1


        curr_target_image_pointer = window_size - 1


        while(curr_target_image_pointer < num_clips_per_vid):
            curr_window_x, curr_window_y = get_current_window_encoder_decoder(data_x, data_y, window_size,
                                                                                curr_target_image_pointer)
            # y_true_i = curr_window_y
            for ii in range(curr_window_x.shape[0]):
                # window_x = data_x[start_ind:end_ind, :, :, :]
                curr_window_x[ii, :, :, :] = (curr_window_x[ii, :, :, :] - cnn_normalization_mean) / cnn_normalization_std



            print('CHECKING WINDOW X AND Y SHAPES')
            print(curr_window_x.shape)
            print(curr_window_y.shape)

            curr_window_x = torch.tensor(curr_window_x, dtype=torch.float)
            # N x Cin x D x H x W
            curr_window_x = curr_window_x.view(1, curr_window_x.shape[1], curr_window_x.shape[0],
                                               curr_window_x.shape[2], curr_window_x.shape[3])

            print(curr_window_x.shape)


            curr_window_y = curr_window_y.view(1, curr_window_y.shape[0], curr_window_y.shape[1], curr_window_y.shape[2])

            # curr_window_x_arr.append(curr_window_x)
            # curr_window_y_arr.append(curr_window_y)

            curr_x_list.append(curr_window_x)
            curr_y_list.append(curr_window_y)

            curr_target_image_pointer += 1

        #     debugging break statement - testing saves and loads:
        # batch_num_debugging_threshold = 10
        # if(batch_num == batch_num_debugging_threshold):
        #     break

    torch_mat_x = torch.squeeze(torch.stack(curr_x_list, dim=0))
    torch_mat_y = torch.squeeze(torch.stack(curr_y_list, dim=0))

    # dataset = Data.TensorDataset(torch_mat_x, torch_mat_y)
    # # BATCH_SIZE_TRAIN = 512
    # # BATCH_SIZE_TRAIN = 1024
    # # BATCH_SIZE_TRAIN = 128
    # train_loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    print('checking torch_mat x and y shapes')
    print(torch_mat_x.shape)
    print(torch_mat_y.shape)
    # sleep(10)
    return torch_mat_x, torch_mat_y



def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)




def train_full_all_batches_windowing_outside_train_harness(encoder_cnn_featurization, encoder, decoder, dnn_model,
                                                            encoder_cnn_featurization_optimizer, encoder_optimizer,
                                                            decoder_optimizer, dnn_model_optimizer, criterion,
                                                            train_loader,
                                                            input_window_size, batch_size_train, iter_num,
                                                           encoder_cnn_feat_file,
                                                           encoder_save_file,
                                                           decoder_save_file, model_save_file):

    encoder_cnn_featurization.train()
    encoder.train()
    decoder.train()
    dnn_model.train()

    # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    # .view the mean and std to make them [C x 1 x 1] so that they can
    # directly work with image Tensor of shape [B x C x H x W].
    # B is batch size. C is number of channels. H is height and W is width

    # encoder_cnn_featurization.cuda()

    input_height = 180
    input_width = 320
    save_every_N_batches = 10

    batch_num = 0
    every_N_print = 1
    total_loss = 0
    # window_size = 4
    window_size = input_window_size
    for step, (torch_window_arr_x, torch_window_arr_y) in enumerate(train_loader):
        print('data:')
        print(torch_window_arr_x.shape)
        # print(false_folder_labels_x)
        print('target')
        print(torch_window_arr_y.shape)
        # print(false_folder_labels_y)
        num_clips_per_vid = torch_window_arr_x.shape[0]

        batch_start_time = time.time()
        if (batch_num % every_N_print == 0):
            print('BATCH NUM:')
            print(batch_num)
            print()
        batch_num += 1
        loss = 0
        encoder_cnn_featurization_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        dnn_model_optimizer.zero_grad()



        torch_window_arr_x = torch_window_arr_x.to(device)
        # torch_window_arr_x = (torch_window_arr_x - cnn_normalization_mean)/cnn_normalization_std

        # torch_window_arr_y = torch_window_arr_y.to(device)
        print('testing before e cnn input')
        print(torch_window_arr_x.shape)
        # print(torch_window_arr_y.shape)

        # torch_window_arr_x_mapped = encoder_cnn_featurization(torch_window_arr_x.cuda())
        torch_window_arr_x_mapped = encoder_cnn_featurization(torch_window_arr_x)

        print('CHECKING SIZE OF ENCODER OUTPUTS')
        # batch_x_lm_full = torch.tensor(batch_x_lm_full, dtype=torch.float)
        # batch_x_full = torch.tensor(batch_x_full, dtype=torch.float)
        # batch_x_full = batch_x_full.to(device)
        # batch_x_lm_full = batch_x_full.to(device)

        # batch_y = batch_y.to(device)

        # print('batch x full size')
        # print(batch_x_full.shape)
        # time.sleep(10)

        torch_window_arr_x_shape = torch_window_arr_x_mapped.shape
        print(torch_window_arr_x_shape)
        print(input_window_size)
        print(encoder.hidden_size)
        encoder_outputs = torch.zeros(torch_window_arr_x_shape[0], input_window_size, encoder.hidden_size, device=device)
        # print('after encoder output init')
        encoder_hidden = encoder.initHidden(torch_window_arr_x_shape)

        #             e_h0, e_c0 = encoder.initHidden()

        #             print(batch_x)
        #         return None, None
        # print('batch x')
        # print(batch_x)
        # print(batch_x.shape)
        # print()
        for ei in range(input_window_size):
            #             print(batch_x[ei].shape)

            #                 encoder_output, encoder_hidden = encoder(batch_x[ei], (e_h0, e_c0))
            # print(ei)

            # batch_x_i = batch_x_full[:, ei]


            # batch_x_i = torch_window_arr_x[:, ei]
            batch_x_i = torch_window_arr_x_mapped[:, ei]


            # batch_x_i = np.array(batch_x_i)

            # print('batch x i:')
            # print(batch_x_i)
            # print(batch_x_i.shape)
            # print(ei)
            # print()

            # batch_x_i = torch.tensor(batch_x_i, dtype=torch.long)
            # batch_x_i = batch_x_i.to(device)

            encoder_output, encoder_hidden = encoder(batch_x_i, encoder_hidden)

            # print('ENCODER OUTPUT')
            # print(encoder_output)
            # print(encoder_hidden)
            # print()

            #                 encoder_outputs[ei] = encoder_output[0, 0]
            encoder_outputs[:, ei, :] = encoder_output
            # print('encoder outputs')
            # print(encoder_outputs)

        # decoder_input = torch.tensor([[SOS_token]*batch_x_full_shape[0]], device=device)
        decoder_input = torch_window_arr_x_mapped
        decoder_hidden = encoder_hidden.squeeze(0)

        print('DECODER INPUT AND HIDDEN INPUT')
        print(decoder_input.shape)
        print(decoder_hidden.shape)
        print('-----------------')

        # batch_y_full = torch.tensor(batch_y_full, dtype=torch.float)
        # batch_y_full = batch_y_full.to(device)

        # free GPU memory:
        # del batch_x_full
        # del batch_x_lm_full

        # map -1 to 0 and 1 to 1:
        # for labels

        # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        y_true_prev = None

        # print('before decoder output training')
        # ------------------------------------------------------
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)


        print('DECODER OUTPUT FULL BATCH TRAINING:')
        print(decoder_output)
        print(decoder_output.shape)
        print(torch_window_arr_x.shape)
        # print(torch_window_arr_x_mapped.shape)
        print()
        decoder_output_shape = decoder_output.shape
        # print(decoder_output.shape)
        # decoder_output = decoder_output.view(decoder_output_shape[0], 1, input_height, input_width)
        # decoder_output = decoder_output.view(decoder_output_shape[0], 3, input_height, input_width)
        decoder_output = decoder_output.view(decoder_output_shape[0], 64, input_height, input_width)
        print(decoder_output.shape)
        target_frame_x = torch.squeeze(torch_window_arr_x[:, :, -1, :, :]).to(device)
        print(target_frame_x.shape)
        # sleep(10)

        y_hat = dnn_model(target_frame_x, decoder_output)
        torch_window_arr_y = torch_window_arr_y.to(device)

        y_true = torch_window_arr_y

        print('before loss, checking decoder output size and y_true shape')
        print(decoder_output.shape)
        print(y_true.shape)
        print('checking shape')



        # FOR LOSS AT LM BRANCH OUTPUT COMBINED WITH LOSS OF NN MODEL:
        # loss = criterion(decoder_output, y_true)
        # loss += criterion(y_hat, y_true)

        # loss = criterion(decoder_output, y_true)
        loss = criterion(y_hat, y_true)

        y_hat_gram = gram_matrix(y_hat)
        y_true_gram = gram_matrix(y_true)

        gram_loss_i = F.mse_loss(y_hat_gram, y_true_gram)

        loss += gram_loss_i

        total_loss += loss

        # print('DEBUGGING: CHECKING SIZE OF Y AND Y HAT:')
        # print(y_hat)
        # print(y)

        del encoder_outputs

        loss.backward()

        # clip grad
        max_grad = 3
        nn.utils.clip_grad_norm(encoder_cnn_featurization.parameters(), max_grad)
        nn.utils.clip_grad_norm(encoder.parameters(), max_grad)
        nn.utils.clip_grad_norm(decoder.parameters(), max_grad)
        nn.utils.clip_grad_norm(dnn_model.parameters(), max_grad)

        encoder_cnn_featurization_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()
        dnn_model_optimizer.step()
        # dnn_model_optimizer.step()
        # encoder_optimizer.step()
        # decoder_optimizer.step()
        delta_t_minutes = (time.time() - batch_start_time) / 60.0
        print('-------------- FULL ELAPSED TIME FOR SINGLE BATCH = %f minutes ------------------' % delta_t_minutes)

        print('DECODER OUTPUT FULL BATCH TRAINING:')
        print(decoder_output)
        print(decoder_output.shape)
        print()

        # save model:
        # print('SAVING MODEL AFTER SINGLE EPOCH TRAINING')
        # save_file_encoder = '/shared/encoder_model_save_file.pth'
        # save_file_decoder = '/shared/decoder_model_save_file.pth'

        # print('BATCH LOSS:')
        # print(loss.item())
        # print('EPOCH NUMBER:')
        # print(iter_num)
        # print()

        loss_curr = loss.item()
        print('BATCH LOSS:')
        print(loss_curr)
        print('BATCH LOSS - AVERAGE LOSS PER SAMPLE:')
        loss_avg_samples_per_batch = loss_curr / (num_clips_per_vid - window_size + 1)
        print(loss_avg_samples_per_batch)
        print('EPOCH NUMBER:')
        print(iter_num)
        print('BATCH NUM 1 INDEXED:')
        print(batch_num)
        print()

        img1_true = y_true[0, :, :, :]
        img1_pred = y_hat[0, :, :, :]
        save_image(img1_true, 'img1_true.png')
        save_image(img1_pred, 'img1_pred.png')

        # saving models every N batches:
        if(batch_num % save_every_N_batches == 0):
            torch.save(dnn_model.state_dict(), model_save_file)
            torch.save(encoder.state_dict(), encoder_save_file)
            torch.save(encoder_cnn_featurization.state_dict(), encoder_cnn_feat_file)
            torch.save(decoder.state_dict(), decoder_save_file)


    # torch.save(encoder, save_file_encoder)
    # torch.save(decoder, save_file_decoder)
    loss_avg_total = total_loss / batch_num
    print('AVG LOSS ACROSS EPOCH - AVG LOSS PER BATCH:')
    print(loss_avg_total)

    return loss_avg_total, encoder_cnn_featurization, encoder, decoder, dnn_model




def test_all_batches_windowing_outside_harness(encoder_cnn_featurization, encoder, decoder, dnn_model,
                                                            train_loader, input_window_size):
    # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)




    criterion_mse = nn.MSELoss()
    avg_psnr = 0
    num_full_samples = 0
    with torch.no_grad():
        y_true_arr = []
        y_pred_arr = []

        encoder_cnn_featurization.eval()
        encoder.eval()
        decoder.eval()
        dnn_model.eval()

        # encoder_cnn_featurization.cuda()

        input_height = 180
        input_width = 320


        batch_num = 0
        every_N_print = 1
        total_loss = 0
        # window_size = 4
        window_size = input_window_size
        # window_size_q = model.max_qlen
        # num_clips_per_vid = train_loader_x.shape[0]

        # for batch_idx, ((data_x, false_folder_labels_x), (data_y, false_folder_labels_y)) in enumerate(
        #         zip(train_loader_x, train_loader_y)):

        for batch_idx, (torch_window_arr_x, torch_window_arr_y) in enumerate(train_loader):
            print('data:')
            print(torch_window_arr_x.shape)
            # print(false_folder_labels_x)
            print('target')
            print(torch_window_arr_y.shape)
            # print(false_folder_labels_y)
            # num_clips_per_vid = torch_window_arr_x.shape[0]

            batch_start_time = time.time()
            if (batch_num % every_N_print == 0):
                print('BATCH NUM:')
                print(batch_num)
                print()
            batch_num += 1
            loss = 0



            torch_window_arr_x = torch_window_arr_x.to(device)
            # torch_window_arr_y = torch_window_arr_y.to(device)
            print('testing before e cnn input')
            print(torch_window_arr_x.shape)
            print(torch_window_arr_y.shape)

            # torch_window_arr_x_mapped = encoder_cnn_featurization(torch_window_arr_x.cuda())
            torch_window_arr_x_mapped = encoder_cnn_featurization(torch_window_arr_x)

            print('CHECKING SIZE OF ENCODER OUTPUTS')

            torch_window_arr_x_shape = torch_window_arr_x_mapped.shape
            print(torch_window_arr_x_shape)
            print(input_window_size)
            print(encoder.hidden_size)
            encoder_outputs = torch.zeros(torch_window_arr_x_shape[0], input_window_size, encoder.hidden_size, device=device)
            # print('after encoder output init')
            encoder_hidden = encoder.initHidden(torch_window_arr_x_shape)


            for ei in range(input_window_size):

                batch_x_i = torch_window_arr_x_mapped[:, ei]

                encoder_output, encoder_hidden = encoder(batch_x_i, encoder_hidden)

                encoder_outputs[:, ei, :] = encoder_output


            # decoder_input = torch.tensor([[SOS_token]*batch_x_full_shape[0]], device=device)
            decoder_input = torch_window_arr_x_mapped
            decoder_hidden = encoder_hidden.squeeze(0)

            print('DECODER INPUT AND HIDDEN INPUT')
            print(decoder_input.shape)
            print(decoder_hidden.shape)
            print('-----------------')


            y_true_prev = None

            # print('before decoder output training')
            # ------------------------------------------------------
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            print('DECODER OUTPUT FULL BATCH TRAINING:')
            print(decoder_output)
            print(decoder_output.shape)
            print(torch_window_arr_x.shape)
            # print(torch_window_arr_x_mapped.shape)
            print()
            decoder_output_shape = decoder_output.shape

            decoder_output = decoder_output.view(decoder_output_shape[0], 64, input_height, input_width)
            print(decoder_output.shape)
            target_frame_x = torch.squeeze(torch_window_arr_x[:, :, -1, :, :]).to(device)
            print(target_frame_x.shape)
            # sleep(10)

            y_hat = dnn_model(target_frame_x, decoder_output)

            mse = criterion_mse(y_hat, torch_window_arr_y.to(device))
            psnr = 10 * log10(1 / mse.item())
            print('CURRENT PSNR:')
            print(psnr)
            avg_psnr += psnr


            # y_true_arr.append()
            y_true = torch_window_arr_y.numpy()
            y_true_arr.append(y_true)
            y_pred_output = y_hat.cpu().numpy()
            y_pred_arr.append(y_pred_output)
            num_full_samples += y_true.shape[0]


            print('before loss, checking decoder output size and y_true shape')
            print(decoder_output.shape)
            print(y_true.shape)
            print('checking shape')

            del encoder_outputs

            delta_t_minutes = (time.time() - batch_start_time) / 60.0
            print('-------------- FULL ELAPSED TIME FOR SINGLE BATCH = %f minutes ------------------' % delta_t_minutes)

            print('DECODER OUTPUT FULL BATCH TRAINING:')
            print(decoder_output)
            print(decoder_output.shape)
            print()

            print('BATCH NUM 1 INDEXED TEST:')
            print(batch_num)
            print()

            img1_true = torch.tensor(y_true[0, :, :, :])
            img1_pred = torch.tensor(y_hat[0, :, :, :])
            save_image(img1_true, 'img1_true_TEST.png')
            save_image(img1_pred, 'img1_pred_TEST.png')


        y_true_arr = np.vstack(y_true_arr)
        # y_pred_arr = np.asarray(y_pred_arr)
        y_pred_arr = np.vstack(y_pred_arr)

    avg_psnr = avg_psnr/num_full_samples
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    return y_true_arr, y_pred_arr, avg_psnr


def train_recurrent_attention_model_windowing_outside_train_harness(encoder_cnn_featurization, encoder,
                                                     decoder, dnn_model, num_epochs, train_loader,input_window_size,
                                                     batch_size_train,
                                                     encoder_cnn_feat_file, encoder_save_file,
                                                     decoder_save_file, model_save_file):

    print_every_const = 1
    plot_every_const = 1


    encoder_cnn_featurization, encoder, \
    attention_decoder, \
    dnn_model = train_num_epochs_windowing_outside_train_harness(encoder_cnn_featurization, encoder, decoder,
                                                                       dnn_model,
                                                                       num_epochs, train_loader,
                                                                       input_window_size, batch_size_train,
                                                                       encoder_cnn_feat_file, encoder_save_file,
                                                                       decoder_save_file, model_save_file,
                                                                       print_every=print_every_const,
                                                                       plot_every=plot_every_const)



    return encoder_cnn_featurization, encoder, attention_decoder, dnn_model




def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def train_num_epochs_windowing_outside_train_harness(encoder_cnn_featurization, encoder, decoder, dnn_model,
                                                           num_epochs, train_loader,
                                                           input_window_size, batch_size_train,
                                                           encoder_cnn_feat_file,
                                                           encoder_save_file,
                                                           decoder_save_file, model_save_file,
                                                           print_every=1000, plot_every=100, learning_rate=0.01):

    start = time.time()
    start_time = start
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    lr_nadam = 1e-4
    dnn_optimizer = Nadam(dnn_model.parameters(), lr=lr_nadam)
    encoder_optimizer = Nadam(encoder.parameters(), lr=lr_nadam)
    encoder_cnn_featurization_optimizer = Nadam(encoder_cnn_featurization.parameters(), lr=lr_nadam)
    decoder_optimizer = Nadam(decoder.parameters(), lr=lr_nadam)

    # dnn_optimizer = QHAdam(dnn_model.parameters(), lr=1e-3, nus=(0.7, 1.0), betas=(0.995, 0.999))
    # encoder_optimizer = QHAdam(encoder.parameters(), lr=1e-3, nus=(0.7, 1.0), betas=(0.995, 0.999))
    # encoder_cnn_featurization_optimizer = QHAdam(encoder_cnn_featurization.parameters(), lr=1e-3, nus=(0.7, 1.0),
    #                                              betas=(0.995, 0.999))
    # decoder_optimizer = QHAdam(decoder.parameters(), lr=1e-3, nus=(0.7, 1.0), betas=(0.995, 0.999))


    # criterion = torch.nn.MSE()
    criterion = torch.nn.SmoothL1Loss()
    n_iters = num_epochs
    for iter in range(1, n_iters + 1):


        loss, encoder_cnn_featurization, encoder, decoder, dnn_model = train_full_all_batches_windowing_outside_train_harness(
            encoder_cnn_featurization, encoder, decoder, dnn_model,
            encoder_cnn_featurization_optimizer, encoder_optimizer,
            decoder_optimizer, dnn_optimizer, criterion,
            train_loader,
            input_window_size, batch_size_train, iter, encoder_cnn_feat_file,
            encoder_save_file,
            decoder_save_file, model_save_file)



        print_loss_total += loss
        plot_loss_total += loss

        # SAVE INTERMEDIATE MODEL AFTER EACH EPOCH:
        # ----------------------------------------------
        torch.save(dnn_model.state_dict(), model_save_file)
        torch.save(encoder.state_dict(), encoder_save_file)
        torch.save(encoder_cnn_featurization.state_dict(), encoder_cnn_feat_file)
        torch.save(decoder.state_dict(), decoder_save_file)


        # ----------------------------------------------

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    delta_t_minutes = (time.time() - start_time) / 60.0
    print('-------------- FULL ELAPSED TIME FOR FULL TRAINING = %f minutes ------------------' % delta_t_minutes)
    # showPlot(plot_losses)
    # create_and_save_plot()
    return encoder_cnn_featurization, encoder, decoder, dnn_model


class rnn_encoder_batch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(rnn_encoder_batch, self).__init__()

        # VARIATIONAL ENCODING:
        # input_dim = 10
        self.fc1 = nn.Linear(hidden_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)




        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.linear = nn.Linear(input_size, hidden_size)
        self.l1_drop = nn.Dropout(p=0.2)

        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)

    #         self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


    def forward(self, input, hidden):

        # VARIATIONAL MODULE:
        embedded = self.linear(input)
        mu, logvar = self.encode(embedded)
        z = self.reparameterize(mu, logvar)
        embedded = self.decode(z)

        embedded = embedded.view(1, input.shape[0], -1)

        print('embedded')
        # print(embedded.cpu().detach().numpy())
        print(embedded.shape)

        output = F.selu(embedded)
        output = self.l1_drop(output)
        # print('before gru')
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self, batch_shape):
        h0 = torch.zeros(1, batch_shape[0], self.hidden_size, device=device)
        nn.init.xavier_normal_(h0)
        c0 = h0
        #         return h0, c0
        return h0



class Encoder_CNN_Featurization(nn.Module):
    def __init__(self, num_chans, num_resid_layers):
        super(Encoder_CNN_Featurization, self).__init__()



        # for SRRESNET SMALLER ARCHITECTURE:
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        # self.num_channels_efficient = 16
        # self.num_channels_efficient = 8
        self.num_channels_efficient = num_chans
        self.input_height = 180
        self.input_width = 320
        full_num_pixels_input = self.input_height * self.input_width
        # full_dim_output_answer_module = int(self.num_channels_efficient * full_num_pixels_input/2)
        full_dim_output_answer_module = int(full_num_pixels_input / 2)

        # out_dim_lin = int(full_dim_output_answer_module/(1e2))
        out_dim_lin = int(full_dim_output_answer_module / (1e3))


        # N x Cin x D x H x W
        self.conv_input = nn.Conv3d(in_channels=3, out_channels=self.num_channels_efficient, kernel_size=9, stride=1,
                                    padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # self.residual = self.make_layer(_Residual_Block, 16)
        # self.residual = self.make_layer(_Residual_Block, 2)
        self.residual = self.make_layer(_Residual_Block_Efficient, self.num_channels_efficient, num_resid_layers)

        # self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_mid = nn.Conv3d(in_channels=self.num_channels_efficient,
                                  out_channels=self.num_channels_efficient, kernel_size=3, stride=1, padding=1,
                                  bias=False)
        # self.bn_mid = nn.InstanceNorm2d(64, affine=True)
        self.bn_mid = nn.InstanceNorm3d(self.num_channels_efficient, affine=True)


    def make_layer(self, block, input_channel_size, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(input_channel_size))
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)

        # out = self.upscale4x(out)
        # out = self.conv_output(out)

        # SRRESNET - OUTPUT
        # BEFORE
        # RESHAPING - STORIES:
        # torch.Size([1, 8, 4, 180, 320])

        # flatten output:
        out_shape = out.shape
        out = out.view(out_shape[0], out_shape[2], 1, -1)

        return out



class rnn_attention_decoder_batch(nn.Module):
    def __init__(self, hidden_size, decoder_input_size_sequence_length, decoder_input_sequence_elem_length,
                 output_size,  max_length, dropout_p=0.1,):
        super(rnn_attention_decoder_batch, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.softsign = nn.Softsign()
        self.tanh = nn.Tanh()
        self.hardtanh = nn.Hardtanh()
        self.sigmoid = nn.Sigmoid()

        # self.decoder_input_size = decoder_input_size
        self.decoder_input_size_sequence_length = decoder_input_size_sequence_length
        self.decoder_input_sequence_elem_length = decoder_input_sequence_elem_length

        # self.embedding = nn.Embedding(self.decoder_input_size, self.hidden_size)
        # self.embedding = nn.Embedding(self.decoder_input_size, 1)

        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.lin_embedding = nn.Linear(decoder_input_sequence_elem_length, hidden_size)
        self.lin_embedding_dropout = nn.Dropout(p=0.2)

        # attn_input_size = hidden_size + self.max_length
        attn_input_size = hidden_size + int(hidden_size * self.max_length)
        # print('attention input size')
        # print(hidden_size)
        # print(max_length)
        # print(attn_input_size)

        self.attn = nn.Linear(attn_input_size, self.max_length)
        self.attn_combine = nn.Linear(attn_input_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        embedded = None
        for i in range(self.decoder_input_size_sequence_length):
            curr_elem = input[:, i, :, :]
            output_lin_embed = F.selu(self.lin_embedding(curr_elem))
            output_lin_embed = self.lin_embedding_dropout(output_lin_embed)
            output_lin_embed = output_lin_embed.squeeze(1)
            if (i == 0):
                embedded = output_lin_embed
            else:

                embedded = torch.cat((embedded, output_lin_embed), dim=1)


        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1)

        attn_weights = attn_weights.view(attn_weights.shape[0], 1, attn_weights.shape[1])

        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        attn_applied = attn_applied.view(attn_applied.shape[0], attn_applied.shape[2])

        output = torch.cat((embedded, attn_applied), 1)

        output = self.attn_combine(output)

        output = F.selu(output)
        output = self.out(output)

        # USE FOR ACTIV FCT FOR OUTPUT WITHOUT INTERMEDIATE CLASS OUTPUT:
        output = self.softsign(output)

        return output, hidden, attn_weights

    def initHidden(self, batch_shape):
        return torch.zeros(1, batch_shape[0], self.hidden_size, device=device)



def get_current_window(data_x, data_y, window_size, window_size_q, curr_target_image_pointer):
    start_ind = curr_target_image_pointer - window_size + 1
    end_ind = curr_target_image_pointer + 1
    window_x = data_x[start_ind:end_ind, :, :, :]

    end_ind_y = end_ind - 1
    # start_ind_y = curr_target_image_pointer - window_size_q + 1 - 1
    start_ind_y = curr_target_image_pointer - window_size_q
    # start_ind_y = end_ind - 2

    window_y = data_y[start_ind_y:end_ind_y, :, :, :]
    return window_x, window_y

