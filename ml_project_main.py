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
from torchvision.utils import save_image

# Parsing helpers
import string
# import nltk
import re
import sys

# device = torch.device("cuda")
device = torch.device("cpu")


# -------------------------------------------------------

# SOS_token = 0
# teacher_forcing_ratio = 0.5
# MAX_LENGTH = 0

from ml_project_data_processing import *
from ml_project_models import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



# TESTING ETL HARNESS
# ---------------------------------------------------------------------------------
if __name__ == '__main__':

    # CHOOSE TARGET DATA:
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # ETL
    desc = 'ETL AND TRAIN/TEST FOR IOT ANALYSIS'

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=desc)

    # parser.add_argument('--run_etl', type=bool, help='Run ETL? Boolean required.', default=True)
    # parser.add_argument('--target_data', type=str, help='Choose unsw or ll', default='unsw')
    # parser.add_argument('--use_weights_class_imbalance', type=str2bool,
    #                     help='Use weights for class imbalance during loss?', default=True)
    parser.add_argument('--use_smaller_train', type=str2bool, help='Use smaller train set?', default=True)
    parser.add_argument('--load_prev_saved_model', type=str2bool, help='Load previously saved model?', default=True)
    parser.add_argument('--run_data_processing', type=str2bool, help='Run data processing? Boolean required.', default=True)

    args = parser.parse_args()
    process_data_bool = args.run_data_processing
    use_smaller_train_bool = args.use_smaller_train
    # run_target_data_str = args.target_data
    # use_weights_for_class_imbalance_bool = args.use_weights_class_imbalance
    # print(use_weights_for_class_imbalance_bool)
    load_prev_saved_model_bool = args.load_prev_saved_model
    # sleep(10)



    start_time = time.time()

    # ETL DATA
    # save_file_prepend = '/data/REDS_data'
    save_file_prepend = '/home/mjs309'
    train_file_path = 'train'
    test_file_path = 'val'
    train_file_path_full = os.path.join(save_file_prepend, train_file_path)
    test_file_path_full = os.path.join(save_file_prepend, test_file_path)
    # train_set, test_set = get_train_test_sets(train_file_path_full, test_file_path_full)


    # PREP DATA:
    # ---------------------------------------------------
    # ---------------------------------------------------
    # ---------------------------------------------------
    model_save_file_prepend = save_file_prepend
    model_save_file = 'dnn_model_save_file.pt'
    encoder_cnn_feat_file = 'encoder_cnn_feat_save_file.pt'
    encoder_save_file = 'encoder_save_file.pt'
    decoder_save_file = 'decoder_save_file.pt'

    model_save_file = os.path.join(model_save_file_prepend, model_save_file)
    encoder_cnn_feat_file = os.path.join(model_save_file_prepend, encoder_cnn_feat_file)
    encoder_save_file = os.path.join(model_save_file_prepend, encoder_save_file)
    decoder_save_file = os.path.join(model_save_file_prepend, decoder_save_file)


    print_every = int(1e1)
    plot_every = int(1e1)
    # model.to(device)

    print('CUDNN ENABLED?')
    print(cudnn.enabled)
    # time.sleep(10)
    # cudnn.enabled = True
    # cudnn.benchmark=True
    use_cuda = torch.cuda.is_available()
    print('USE CUDA?')
    print(use_cuda)

    # num_frames_per_vid_clip = 100
    # num_frames_per_vid_clip = 10
    num_frames_per_vid_clip = 5
    # num_frames_per_vid_clip = 25
    # input_window_size_x = 4
    input_window_size_x = 4
    # batch_size = 40
    # batch_size = 5
    batch_size = 2

    MAX_LENGTH = input_window_size_x
    # hidden_size = 256
    # hidden_size = 32
    hidden_size = 32

    input_height = 180
    input_width = 320
    # NUM_EPOCHS = 10
    NUM_EPOCHS = int(1e1)
    # NUM_EPOCHS = int(2e0)
    # NUM_EPOCHS = int(3e0)
    # NUM_EPOCHS = 0
    dropout_prob = 0.2
    num_channels_encoder_featurization = 2
    num_resid_layers_encoder_featurization = 1
    full_num_pixels_input = input_height * input_width
    # full_dim_output_answer_module = int(self.num_channels_efficient * full_num_pixels_input/2)
    full_dim_output_answer_module = int(full_num_pixels_input / 2)
    # mult_factor_decoder_output = 3
    mult_factor_decoder_output = 64

    # decoder_output_size = full_num_pixels_input
    decoder_output_size = full_num_pixels_input*mult_factor_decoder_output


    num_channels_srresnet = 64
    # num_resid_layers_srresnet = 16
    num_resid_layers_srresnet = 4


    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()



    torch_arr_x_train_file = 'torch_arr_x_train_file.pt'
    torch_arr_y_train_file = 'torch_arr_y_train_file.pt'

    torch_arr_x_train_smaller_file = 'torch_arr_x_train_smaller_file.pt'
    torch_arr_y_train_smaller_file = 'torch_arr_y_train_smaller_file.pt'

    torch_arr_x_test_file = 'torch_arr_x_test_file.pt'
    torch_arr_y_test_file = 'torch_arr_y_test_file.pt'

    # numpy
    numpy_arr_x_train_file = 'numpy_arr_x_train_file.npy'
    numpy_arr_y_train_file = 'numpy_arr_y_train_file.npy'

    numpy_arr_x_train_smaller_file = 'numpy_arr_x_train_smaller_file.npy'
    numpy_arr_y_train_smaller_file = 'numpy_arr_y_train_smaller_file.npy'

    numpy_arr_x_test_file = 'numpy_arr_x_test_file.npy'
    numpy_arr_y_test_file = 'numpy_arr_y_test_file.npy'

    saved_data_dir_name = 'saved_data_processing_variables'


    torch_arr_x_train_file = os.path.join(save_file_prepend, saved_data_dir_name, torch_arr_x_train_file)
    torch_arr_y_train_file = os.path.join(save_file_prepend, saved_data_dir_name, torch_arr_y_train_file)

    torch_arr_x_train_smaller_file = os.path.join(save_file_prepend, saved_data_dir_name, torch_arr_x_train_smaller_file)
    torch_arr_y_train_smaller_file = os.path.join(save_file_prepend, saved_data_dir_name, torch_arr_y_train_smaller_file)

    torch_arr_x_test_file = os.path.join(save_file_prepend, saved_data_dir_name, torch_arr_x_test_file)
    torch_arr_y_test_file = os.path.join(save_file_prepend, saved_data_dir_name, torch_arr_y_test_file)



    # numpy:
    numpy_arr_x_train_file = os.path.join(save_file_prepend, saved_data_dir_name, numpy_arr_x_train_file)
    numpy_arr_y_train_file = os.path.join(save_file_prepend, saved_data_dir_name, numpy_arr_y_train_file)

    numpy_arr_x_train_smaller_file = os.path.join(save_file_prepend, saved_data_dir_name,
                                                  numpy_arr_x_train_smaller_file)
    numpy_arr_y_train_smaller_file = os.path.join(save_file_prepend, saved_data_dir_name,
                                                  numpy_arr_y_train_smaller_file)

    numpy_arr_x_test_file = os.path.join(save_file_prepend, saved_data_dir_name, numpy_arr_x_test_file)
    numpy_arr_y_test_file = os.path.join(save_file_prepend, saved_data_dir_name, numpy_arr_y_test_file)

    print('checking process data and smaller train bools')
    print(process_data_bool)
    print(type(process_data_bool))
    print(use_smaller_train_bool)
    print(type(use_smaller_train_bool))
    sleep(10)
    # train_loader_x = None
    # train_loader_y = None
    # test_loader_x = None
    # test_loader_y = None

    if(process_data_bool):

        print('PULLING IMAGES FROM DATA DIRECTORIES')
        X_list_train, Y_list_train, X_list_test, Y_list_test = get_train_test_sets(train_file_path_full,
                                                                                   test_file_path_full)

        train_loader_x = Data.DataLoader(
            X_list_train,
            batch_size=num_frames_per_vid_clip,
            # num_workers=1,
            shuffle=False
        )
        train_loader_y = Data.DataLoader(
            Y_list_train,
            batch_size=num_frames_per_vid_clip,
            # num_workers=1,
            shuffle=False
        )

        test_loader_x = Data.DataLoader(
            X_list_test,
            batch_size=num_frames_per_vid_clip,
            # num_workers=1,
            shuffle=False
        )
        test_loader_y = Data.DataLoader(
            Y_list_test,
            batch_size=num_frames_per_vid_clip,
            # num_workers=1,
            shuffle=False
        )

        print('CREATING X AND Y TRAIN AND TEST DATASETS')
        torch_arr_x_train, torch_arr_y_train = create_X_and_Y(train_loader_x, train_loader_y, input_window_size_x)
        torch_arr_x_test, torch_arr_y_test = create_X_and_Y(test_loader_x, test_loader_y, input_window_size_x)

        del (train_loader_x)
        del (train_loader_y)
        del (test_loader_x)
        del (test_loader_y)

        #     save data tensors:
        # torch.save(tensor, 'file.pt') and torch.load('file.pt')
        torch.save(torch_arr_x_train, torch_arr_x_train_file)
        torch.save(torch_arr_y_train, torch_arr_y_train_file)
        torch.save(torch_arr_x_test, torch_arr_x_test_file)
        torch.save(torch_arr_y_test, torch_arr_y_test_file)


    else:
        #     load data tensors
        print('LOAD X AND Y TRAIN AND TEST DATASETS')

        print('LOAD X TEST DATASETS')
        # torch_arr_x_test = torch.load(torch_arr_x_test_file)
        numpy_arr_x_test = np.load(numpy_arr_x_test_file)
        torch_arr_x_test = torch.from_numpy(numpy_arr_x_test)

        print('LOAD Y TEST DATASETS')
        # torch_arr_y_test = torch.load(torch_arr_y_test_file)
        numpy_arr_y_test = np.load(numpy_arr_y_test_file)
        torch_arr_y_test = torch.from_numpy(numpy_arr_y_test)


        num_samples_test = torch_arr_x_test.shape[0]
        print('NUM TEST SAMPLES')
        print(num_samples_test)

        # # save test as numpy
        # numpy_arr_x_test = torch_arr_x_test.numpy()
        # # np to torch:
        # # torch_arr_x_test = torch.from_numpy(numpy_arr_x_test)
        # numpy_arr_y_test = torch_arr_y_test.numpy()
        #
        # np.save(numpy_arr_x_test_file, numpy_arr_x_test)
        # np.save(numpy_arr_y_test_file, numpy_arr_y_test)


        # # save smaller version of train set:
        # print('CREATING SMALLER TRAIN LOAD FILES')
        # torch_arr_x_train_smaller = torch_arr_x_train[:num_samples_test, :, :, :, :]
        # torch_arr_y_train_smaller = torch_arr_y_train[:num_samples_test, :, :, :]
        #
        # del(torch_arr_x_train)
        # del(torch_arr_y_train)
        #
        # torch.save(torch_arr_x_train_smaller, torch_arr_x_train_smaller_file)
        # torch.save(torch_arr_y_train_smaller, torch_arr_y_train_smaller_file)
        # print('CREATING SMALLER TRAIN LOAD FILES COMPLETE


        if(use_smaller_train_bool):
            print('LOAD X SMALLER TRAIN DATASETS')
            # torch_arr_x_train = torch.load(torch_arr_x_train_smaller_file)
            numpy_arr_x_train = np.load(numpy_arr_x_train_file)
            torch_arr_x_train = torch.from_numpy(numpy_arr_x_train)


            print('LOAD Y SMALLER TRAIN DATASETS')
            # torch_arr_y_train = torch.load(torch_arr_y_train_smaller_file)
            numpy_arr_y_train = np.load(numpy_arr_y_train_file)
            torch_arr_y_train = torch.from_numpy(numpy_arr_y_train)

            # # save test as numpy
            # numpy_arr_x_train = torch_arr_x_train.numpy()
            # # np to torch:
            # # torch_arr_x_test = torch.from_numpy(numpy_arr_x_test)
            # numpy_arr_y_train = torch_arr_y_train.numpy()
            #
            # # np.save(numpy_arr_x_train_file, numpy_arr_x_train)
            # # np.save(numpy_arr_y_train_file, numpy_arr_y_train)
            #
            # numpy_arr_x_test = np.load(numpy_arr_x_test_file)
            # numpy_arr_y_test = np.load(numpy_arr_y_test_file)


        else:
            print('LOAD X TRAIN DATASETS')
            torch_arr_x_train = torch.load(torch_arr_x_train_file)
            # torch_arr_x_train = torch.load(torch_arr_x_train_smaller_file)
            print('LOAD Y TRAIN DATASETS')
            torch_arr_y_train = torch.load(torch_arr_y_train_file)
            # torch_arr_y_train = torch.load(torch_arr_y_train_smaller_file)

            # # save test as numpy
            # numpy_arr_x_test = torch_arr_x_test.numpy()
            # # np to torch:
            # # torch_arr_x_test = torch.from_numpy(numpy_arr_x_test)
            # numpy_arr_y_test = torch_arr_y_test.numpy()
            #
            # np.save(numpy_arr_x_test_file, numpy_arr_x_test)
            # np.save(numpy_arr_y_test_file, numpy_arr_y_test)
            #
            # # numpy_arr_x_test = np.load(numpy_arr_x_test_file)
            # # numpy_arr_y_test = np.load(numpy_arr_y_test_file)

    print('LOADING COMPLETE')


    # del (train_loader_x)
    # del (train_loader_y)
    # del (test_loader_x)
    # del (test_loader_y)


    def create_loader(x, y, batch_size):
        dataset = Data.TensorDataset(x, y)
        # BATCH_SIZE_TRAIN = 512
        # BATCH_SIZE_TRAIN = 1024
        # BATCH_SIZE_TRAIN = 128
        # loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return loader

    train_loader = create_loader(torch_arr_x_train, torch_arr_y_train, batch_size)

    del(torch_arr_x_train)
    del(torch_arr_y_train)

    test_loader = create_loader(torch_arr_x_test, torch_arr_y_test, batch_size)

    del(torch_arr_x_test)
    del(torch_arr_y_test)



    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # e_cnn_feat_model = encoder_cnn_featurization(num_channels_encoder_featurization, num_resid_layers_encoder_featurization)
    encoder_cnn_featurization = Encoder_CNN_Featurization(num_channels_encoder_featurization, num_resid_layers_encoder_featurization)




    # embed_size, decoder_input_size_sequence_length, decoder_input_sequence_elem_length = get_encoder_and_decoder_input_sizes(X_feat_lm_data)
    embed_size, decoder_input_size_sequence_length,\
    decoder_input_sequence_elem_length = get_encoder_and_decoder_input_sizes_loader(train_loader, encoder_cnn_featurization, input_window_size_x)
    # decoder_input_sequence_elem_length = get_encoder_and_decoder_input_sizes(train_loader_x, encoder_cnn_featurization, input_window_size_x)

    # embed_size = np.array(x).max() + 1
    print('embed_size and decoder input size and sequence elem size')
    print(embed_size)
    print(decoder_input_size_sequence_length)
    print(decoder_input_sequence_elem_length)


    # TRAIN AND TEST MODEL:
    # ----------------------------------------------------------------
    start_train_test_time = time.time()

    # encoder = rnn_encoder(embed_size, hidden_size).to(device)
    input_size_for_encoder = embed_size
    encoder = rnn_encoder_batch(input_size_for_encoder, hidden_size).to(device)
    # encoder = rnn_encoder_batch_softmax(embed_size, hidden_size).to(device)

    print('ENCODER MODEL')
    print(encoder)

    print('DECODER OUTPUT and ENCODER AND DECODER INPUT SIZES')
    print('--------------------------')
    print(embed_size)
    print(decoder_output_size)
    # print(x)
    # print(x.shape)
    # print(y)
    print('--------------------------')
    # sl(5)
    # attention_decoder = rnn_attention_decoder_batch_softmax(hidden_size, decoder_input_size, decoder_output_size, dropout_p=dropout_prob).to(device)
    attention_decoder = rnn_attention_decoder_batch(hidden_size, decoder_input_size_sequence_length,
                                                                 decoder_input_sequence_elem_length,
                                                                 decoder_output_size, MAX_LENGTH,
                                                                 dropout_p=dropout_prob).to(device)

    # dnn_model = SRResNet()
    # # num_channels_srresnet = 8
    # num_channels_srresnet = 8
    # # num_resid_layers_srresnet = 2
    # num_resid_layers_srresnet = 2
    dnn_model = SRResNet(num_channels_srresnet, num_resid_layers_srresnet)
    dnn_model.apply(weights_init)
    encoder.apply(weights_init)
    encoder_cnn_featurization.apply(weights_init)
    attention_decoder.apply(weights_init)

    if (load_prev_saved_model_bool):
        # LOAD previous MODEL:
        dnn_model.load_state_dict(torch.load(model_save_file))
        encoder_cnn_featurization.load_state_dict(torch.load(encoder_cnn_feat_file))
        encoder.load_state_dict(torch.load(encoder_save_file))
        attention_decoder.load_state_dict(torch.load(decoder_save_file))

    # TRAINING
    # -----------------------------------------------------------------------------

    dnn_model.to(device)
    encoder_cnn_featurization.to(device)
    encoder.to(device)
    attention_decoder.to(device)

    # batch_size_train = 100
    batch_size_train = num_frames_per_vid_clip

    encoder_cnn_featurization, encoder, \
    attention_decoder, \
    dnn_model = train_recurrent_attention_model_windowing_outside_train_harness(encoder_cnn_featurization, encoder,
                                                                 attention_decoder, dnn_model, NUM_EPOCHS,
                                                                 train_loader, input_window_size_x,
                                                                 batch_size_train,
                                                                 encoder_cnn_feat_file, encoder_save_file,
                                                                 decoder_save_file, model_save_file)


    # SAVE FINAL MODEL:
    # dnn_model.save_state_dict(dnn_model_save_file)
    # encoder.save_state_dict(encoder_model_save_file)
    # attention_decoder.save_state_dict(decoder_model_save_file)

    torch.save(dnn_model.state_dict(), model_save_file)
    torch.save(encoder.state_dict(), encoder_save_file)
    torch.save(encoder_cnn_featurization.state_dict(), encoder_cnn_feat_file)
    torch.save(attention_decoder.state_dict(), decoder_save_file)



    print('FINISHED TRAINING - STARTING TESTING')
    y_true_arr, y_pred_arr, avg_psnr = test_all_batches_windowing_outside_harness(encoder_cnn_featurization, encoder,
                                                                                   attention_decoder, dnn_model,
                                                                                    test_loader,
                                                                                   input_window_size_x)


    y_true_arr_save_file = 'y_true_arr_save_file.npy'
    y_pred_arr_save_file = 'y_pred_arr_save_file.npy'
    np.save(y_true_arr_save_file, y_true_arr)
    np.save(y_pred_arr_save_file, y_pred_arr)
    print('TORCH CUDA VERSION:')
    print(torch.version.cuda)
    print('------------------TRAIN AND TEST DONE----------------------------')

    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))

    # delta_time_run = (time.time() - start_time) / 60.0
    # print('--------------FULL RUNTIME OF ETL TEST HARNESS TOOK %f minutes --------------' % delta_time_run)



    delta_time_run = (time.time() - start_time) / 60.0
    print('--------------FULL RUNTIME OF ETL TEST HARNESS TOOK %f minutes --------------' % delta_time_run)













