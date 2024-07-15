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

# Parsing helpers
import string
# import nltk
import re
import sys

# %matplotlib inline
from graphviz import Digraph

# device = torch.device("cuda")
device = torch.device("cpu")


def get_train_test_sets(train_file_path, test_file_path):
    low_res_train_dir = 'train_sharp_bicubic'
    hi_res_train_dir = 'train_sharp'
    low_res_test_dir = 'val_sharp_bicubic'
    hi_res_test_dir = 'val_sharp'

    low_res_train_dir_path = os.path.join(train_file_path, low_res_train_dir)
    hi_res_train_dir_path = os.path.join(train_file_path, hi_res_train_dir)

    low_res_test_dir_path = os.path.join(test_file_path, low_res_test_dir)
    hi_res_test_dir_path = os.path.join(test_file_path, hi_res_test_dir)

    # one tensor for each vid clip (each vid clip is a folder of 100 images)
    X_list_train = low_res_train_list_of_tensors = get_list_of_video_tensors_from_target_dir(low_res_train_dir_path)
    Y_list_train = hi_res_train_list_of_tensors = get_list_of_video_tensors_from_target_dir(hi_res_train_dir_path)

    X_list_test = low_res_test_list_of_tensors = get_list_of_video_tensors_from_target_dir(low_res_test_dir_path)
    Y_list_test = hi_res_test_list_of_tensors = get_list_of_video_tensors_from_target_dir(hi_res_test_dir_path)

    return X_list_train, Y_list_train, X_list_test, Y_list_test



def get_list_of_video_tensors_from_target_dir(target_dir_full_path):
    count = 0
    tensor_list = []

    tensor_list = get_tensor_for_target_dir(target_dir_full_path)

    # for dir_i in sorted(os.listdir(target_dir_full_path)):
    #     full_f_i_path = os.path.join(target_dir_full_path, dir_i)
    #     tensor_i = get_tensor_for_target_dir(full_f_i_path)
    #     tensor_list.append(tensor_i)

    return tensor_list


def get_tensor_for_target_dir(target_dir):
    train_set_i = torchvision.datasets.ImageFolder(root=target_dir, transform=torchvision.transforms.ToTensor())
    return train_set_i








# def create_full_matrix_from_files_dir(target_dir_full_path):
#     count = 0
#     full_matrix = None
#     for file_i in sorted(os.listdir(target_dir_full_path)):
#         full_f_i_path = os.path.join(target_dir_full_path, file_i)
#         print(full_f_i_path)
#         if(count == 0):
#             # full_matrix = np.load(full_f_i_path)
#             full_matrix = [np.load(full_f_i_path)]
#
#         else:
#             curr_mat = np.load(full_f_i_path)
#             # full_matrix = np.vstack((full_matrix, curr_mat))
#             full_matrix.append(curr_mat)
#
#         count += 1
#
#     full_matrix = np.vstack((full_matrix))
#     return full_matrix


















