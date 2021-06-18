import argparse
import torchvision
import os

parser = argparse.ArgumentParser(description="Data preparation for training Cifar10")
parser.add_argument("--save_folder", type=str, help='Folder to save data after download',
                    default="/home/tan.le2/PycharmProjects/cifar10_kubeflow/pytorch-cifar/data")
args = parser.parse_args()

save_folder = args.save_folder
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

trainset = torchvision.datasets.CIFAR10(root=save_folder, train=True, download=True)
testset = torchvision.datasets.CIFAR10(root=save_folder, train=False, download=True)