import os
import copy
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
# from modules import UNet_conditional, EMA
from diffusion_modules import UNetStandard
import logging
from torch.utils.tensorboard import SummaryWriter
import time

from torchvision import models
from torchsummary import summary

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

from time import sleep

import math
from math import log10, sqrt

def print_size_of_model(model):
    print('printing size of model:')
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('number of parameters: {:.3f}'.format(param_size))
    print('model size: {:.3f}MB'.format(size_all_mb))
    print()


def get_amount_free_memory_during_train():
    t = torch.cuda.get_device_properties(0).total_memory
    # r = torch.cuda.memory_reserved(0)
    # a = torch.cuda.memory_allocated(0)
    r = torch.cuda.memory_reserved()
    a = torch.cuda.memory_allocated()
    f = r - a  # free inside reserved
    f_mb = f / 1024 ** 2
    print("IN TRAIN LOOP - CURRENT FREE MEMORY AMOUNT = {:.3f}".format(f_mb))


def sample_images_UNet(model, input_images, device):
    model.eval()
    with torch.no_grad():
        x = input_images.to(device)

        predicted_images = model(x)
        predicted_images.detach().to('cpu')
        predicted_images_no_uint8_mapping = predicted_images

        predicted_images = (predicted_images.clamp(-1, 1) + 1) / 2
        predicted_images = (predicted_images * 255).type(torch.uint8)
        del(x)
        del(input_images)

    model.train()
    return predicted_images, predicted_images_no_uint8_mapping



def get_img_width_height(dataloader):
    for i, (images_x, labels_x) in enumerate(dataloader):
        print('checking for img width and height')
        print(images_x.shape)
        img_width = images_x.shape[3]
        img_height = images_x.shape[2]
        break
    return img_width, img_height

def PSNR_Avg(batchorg, batchtest):
    psnr_avg_value = 0
    for i in range(batchtest.shape[0]):
        psnr_current = PSNR(batchorg[i, :].numpy(),batchtest[i, :].numpy())
        psnr_avg_value += psnr_current
    psnr_avg_value /= batchtest.shape[0]
    return psnr_avg_value


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def train(args):
    print('STARTING TRAINING')
    setup_logging(args.run_name)
    device = args.device
    dataloader_x, dataloader_y = get_data_x_and_y(args)

    hidden_channel_dims = args.hidden_channel_dims
    use_ESN = args.use_ESN
    img_width, img_height = get_img_width_height(dataloader_y)


    model = UNetStandard(device=device, hidden_channel_dims=hidden_channel_dims,
                         # use_simple_linear_mapping=use_simple_linear_mapping,
                         use_ESN=use_ESN,
                         reservoir_size_multiplier=args.reservoir_size_multiplier,
                         target_img_width=img_width, target_img_height=img_height
                        ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader_x)
    sample_every_N = args.sample_every_N

    print_size_of_model(model)

    print('SUMMARY OF MODEL')
    summary(model, (3, 180, 320))
    print()
    sleep(5)
    psnr_array = []

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader_x)
        dataloader_y_iterator = iter(dataloader_y)



        for i, (images_x, labels_x) in enumerate(pbar):

            try:
                images_y, labels_y = next(dataloader_y_iterator)
            except StopIteration:
                dataloader_y_iterator = iter(dataloader_y)
                images_y, labels_y = next(dataloader_iterator)


            images_x = images_x.to(device)

            predicted_images_y = model(images_x)
            loss = mse(images_y.to(device), predicted_images_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.detach().item())
            logger.add_scalar("MSE", loss.detach().item(), global_step=epoch * l + i)

            # print('BEFORE VARIABLES CLEARED FOR GPU SPACE')
            # print('CUDA MEMORY ALLOCATED:')
            # print(torch.cuda.memory_allocated() // 1024**2)
            # print()

            images_x.detach().to('cpu')
            images_y.detach().to('cpu')
            loss.detach().to('cpu')
            predicted_images_y.detach().to('cpu')
            # del(images_x)

            # del(images_y)

            del(loss)
            del(predicted_images_y)
            # torch.cuda.empty_cache()


            # get_amount_free_memory_during_train()


            # print('CUDA MEMORY ALLOCATED:')
            # print(torch.cuda.memory_allocated() // 1024**2)

            # sleep(5)
            # torch.cuda.empty_cache()

            ## below break for debugging:
            # break

        #showing image prediction capability at epoch N:
        # ------------------------------------------------------------
        if epoch % sample_every_N == 0:
            sampled_images, predicted_images_no_uint8_mapping = sample_images_UNet(model, images_x, device)
            predicted_images_no_uint8_mapping = predicted_images_no_uint8_mapping.detach().to('cpu')

            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))

            PSNR_value = PSNR_Avg(images_y, predicted_images_no_uint8_mapping)
            print('------------------')
            print(f"PSNR value is {PSNR_value} dB")
            # print(PSNR(images_y, predicted_images_no_uint8_mapping))
            print('------------------')
            psnr_array.append(PSNR_value)


            ##save model checkpoint - ckpt for short:
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

            ## save optimizer - optim for short
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
        # ------------------------------------------------------------
        ## below break for debugging:
        # break

    x = np.arange(1, len(psnr_array)+1)
    plt.title("PSNR for epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR value")
    plt.plot(x, psnr_array)
    plt.show()





def test(args):
    print('STARTING TESTING')

    setup_logging_for_test(args.run_name)
    # rand_sample_p = 0.1
    rand_sample_p = 1.0
    device = args.device
    dataloader_x, dataloader_y = get_data_x_and_y(args, bTraining=False)

    hidden_channel_dims = args.hidden_channel_dims
    use_ESN = args.use_ESN
    mse = nn.MSELoss()


    img_width, img_height = get_img_width_height(dataloader_y)

    model = UNetStandard(device=device, hidden_channel_dims=hidden_channel_dims,
                         # use_simple_linear_mapping=use_simple_linear_mapping,
                         use_ESN=use_ESN,
                         reservoir_size_multiplier=args.reservoir_size_multiplier,
                         target_img_width=img_width, target_img_height=img_height
                        ).to(device)

    ckpt = torch.load("./models/UNetHighRes/ckpt.pt")
    # ckpt = torch.load("./models_linearMapping_saved_Jan11_2023/UNetHighRes/ckpt.pt")
    model.load_state_dict(ckpt)

    print_size_of_model(model)

    print('SUMMARY OF MODEL')
    summary(model, (3, 180, 320))
    print()
    sleep(3)

    loss = 0
    loss_batch_count = 0

    pbar = tqdm(dataloader_x)
    dataloader_y_iterator = iter(dataloader_y)
    for i, (images_x, labels_x) in enumerate(pbar):

        try:
            images_y, labels_y = next(dataloader_y_iterator)
        except StopIteration:
            dataloader_y_iterator = iter(dataloader_y)
            images_y, labels_y = next(dataloader_iterator)

        images_x = images_x.to(device)

        if np.random.random() < rand_sample_p:
            predicted_images_y = sample_and_save_output_images_testing(images_x, model, args, i)

            loss += mse(images_y.to(device), predicted_images_y).item()
            # print(loss)
            loss_batch_count += 1


    loss_avg = loss / loss_batch_count
    print('Average Batch MSE Loss on Validation Dataset')
    print(loss_avg)
    print()
    sleep(4)




def sample_and_save_output_images(input_images, model, args):
    device = args.device
    sampled_images, predicted_images_no_uint8_mapping = sample_images_UNet(model, input_images, device)
    plot_images(sampled_images)
    save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))

    ##save model checkpoint - ckpt for short:
    torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

    ## save optimizer - optim for short
    torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

    return predicted_images_no_uint8_mapping




def sample_and_save_output_images_testing(input_images, model, args, batch_num):
    device = args.device
    sampled_images, predicted_images_no_uint8_mapping = sample_images_UNet(model, input_images, device)

    plot_images(sampled_images)
    save_images(sampled_images, os.path.join("./test_results", args.run_name, f"{batch_num}.jpg"))
    return predicted_images_no_uint8_mapping




def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "UNetHighRes"

    args.train = True
    # args.train = False
    args.test = True
    # args.test = False

    ## if true, uses ESN block for final layer to output result
    ## else, uses simple linear mapping and convolution for final layer to output result
    args.use_ESN = True
    # args.use_ESN = False

    # args.epochs = 300
    # args.epochs = 10
    args.epochs = 11
    print('Num epochs for train:')
    print(args.epochs)
    print()

    # args.sample_every_N = 2
    args.sample_every_N = 1

    # original was batch size 4
    # args.batch_size = 1
    args.batch_size = 2
    # args.batch_size = 4
    # args.batch_size = 14
    # args.batch_size = 32
    #testing my own here
    # args.batch_size = 6

    #original was multiplier 100
    args.reservoir_size_multiplier = 100
    # args.reservoir_size_multiplier = 10
    # args.reservoir_size_multiplier = 20

    #original was 8
    args.hidden_channel_dims = 16
    # args.hidden_channel_dims = 8
    # args.hidden_channel_dims = 4

    print('THIS RUN USED BATCH SIZE 2, RESERVOIR OF 100, AND 16 HIDDEN DIM UNITS')

    using_LNN_str = 'USING LIQUID NEURAL NETWORK - ESN BLOCK = ' + str(args.use_ESN)
    print(using_LNN_str)
    print('NOTE, IF THE ABOVE IS FALSE, THEN MODEL WILL USE STANDARD UNET LINEAR MAPPING FOR OUTPUT')
    training_str = "Training = " + str(args.train)
    testing_str = "Testing = " + str(args.test)
    print(training_str)
    print(testing_str)
    print()
    sleep(10)

    # train set:
    # args.dataset_path_x = "/shared/marc_ML_project/Datasets/REDS/data_concise/train/train_sharp_bicubic"
    # args.dataset_path_y = "/shared/marc_ML_project/Datasets/REDS/data_concise/train/train_sharp"
    # args.dataset_path_x = "/datasets/REDS/data_concise/train/train_sharp_bicubic"
    # args.dataset_path_y = "/datasets/REDS/data_concise/train/train_sharp"
    args.dataset_path_x = "/shared/train/train_sharp_bicubic"
    args.dataset_path_y = "/shared/train/train_sharp"


    # validation set:
    # args.test_dataset_path_x = "/shared/marc_ML_project/Datasets/REDS/data_concise/val/val_sharp_bicubic"
    # args.test_dataset_path_y = "/shared/marc_ML_project/Datasets/REDS/data_concise/val/val_sharp"
    # args.test_dataset_path_x = "/datasets/REDS/data_concise/val/val_sharp_bicubic"
    # args.test_dataset_path_y = "/datasets/REDS/data_concise/val/val_sharp"
    args.test_dataset_path_x = "/shared/val/val_sharp_bicubic"
    args.test_dataset_path_y = "/shared/val/val_sharp"

    args.device = "cuda"
    # args.device = "cpu"
    args.lr = 3e-4


    if args.train:
        train(args)

    if args.test:
        test(args)





if __name__ == '__main__':
    start_time = time.time()
    launch()

    delta_time_run = (time.time() - start_time) / 60.0
    print('--------------FULL RUNTIME OF UNET TRAIN/TEST HARNESS TOOK %f minutes --------------' % delta_time_run)


