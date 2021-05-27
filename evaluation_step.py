import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from resnet_models import ResNet18
import argparse

torch.manual_seed(43)

parser = argparse.ArgumentParser(description="Pytorch CIFAR10 training")

parser.add_argument('--batch_size', type=int, default=32, help="batch size use when training")
parser.add_argument('--model', type=float, default=0.01, help="init learningrate")
parser.add_argument("--input_data", type=str, default="/home/tan.le2/PycharmProjects/cifar10_kubeflow/pytorch-cifar/data",
                    help="Directory include data")
args = parser.parse_args()


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(root=args.input_data, download=False, train=False,
                                              transform=transform_test)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print("Creating model")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(torch.load("./checkpoint/ckpt.pth")['net'])

def evaluation_model():

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print("Test acc: ", acc)

evaluation_model()
