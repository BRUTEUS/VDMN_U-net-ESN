import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
    # plt.savefig('foo.png')


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)



def get_data_x_and_y(args, bTraining=True):
    if bTraining:
        dataset_path_x = args.dataset_path_x
        dataset_path_y = args.dataset_path_y

    else:
        dataset_path_x = args.test_dataset_path_x
        dataset_path_y = args.test_dataset_path_y

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_x = torchvision.datasets.ImageFolder(dataset_path_x, transform=transforms)
    dataset_y = torchvision.datasets.ImageFolder(dataset_path_y, transform=transforms)

    dataloader_x = DataLoader(dataset_x, batch_size=args.batch_size, shuffle=False)
    dataloader_y = DataLoader(dataset_y, batch_size=args.batch_size, shuffle=False)
    return dataloader_x, dataloader_y



def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def setup_logging_for_test(run_name):
    os.makedirs("test_results", exist_ok=True)
    os.makedirs(os.path.join("test_results", run_name), exist_ok=True)
