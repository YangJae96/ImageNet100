import torch
import torch.nn as nn
import time
import os
import argparse

from config import cfg
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

parser = argparse.ArgumentParser(description='ImageNet-100 Classification')
parser.add_argument('--base_path', default=os.getcwd(),
                    help='default path')
parser.add_argument('--model', default=False,
                    help='Model Architecture Name(AlexNet, ResNet18, ResNet50, DenseNet161')
parser.add_argument('--model_path', default=False,
                    help='Model Path for test')

args = parser.parse_args()
path=args.base_path
model_path = args.model_path

USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if torch.cuda.is_available():
    device=torch.device('cuda')

# Constant to control how frequently we print train loss
print('using device:', device)
print(torch.cuda.get_device_name())

simple_transform=transforms.Compose([transforms.Resize((224,224))
                                    ,transforms.ToTensor()
                                    ,transforms.Normalize(cfg.pixel_mean,
                                                          cfg.pixel_std)])
print("Loading ImageNet-100 Test Dataset.....", end=" ")

test=ImageFolder(cfg.test_dir,simple_transform)
test_data_gen=torch.utils.data.DataLoader(test, batch_size=cfg.test_batch_size, num_workers=cfg.num_workers)

def load_model(model_name):
    if model_name=="AlexNet":
        print("Loading AlexNet Model")
        model_ft = models.alexnet()
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 100)
        model_ft.load_state_dict(torch.load(cfg.test_model_path))
    elif model_name == "ResNet18":
        print("Loading ResNet18 Model")
        model_ft = models.resnet18()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 100)
        model_ft.load_state_dict(torch.load(cfg.test_model_path))
    elif model_name=="ResNet50":
        print("Loading ResNet50 Model")
        model_ft = models.resnet50()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 100)
        model_ft.load_state_dict(torch.load(cfg.test_model_path))
    elif model_name=="DenseNet":
        print("Loading DenseNet161 Model")
        model_ft = models.densenet161()
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 100)
        model_ft.load_state_dict(torch.load(cfg.test_model_path))

    return model_ft

def validation(model, testloader, criterion):
    model = model.to(device=device, dtype=dtype)
    model.train(False)
    
    run_loss = 0
    run_corrects = 0

    for inputs, labels in testloader:
    
        if torch.cuda.is_available():
            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.long)
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)

        run_loss += loss.data
        run_corrects += torch.sum(preds == labels.data)

        epoch_loss = run_loss.item() / len(testloader.dataset)
        epoch_acc = run_corrects.item() / len(testloader.dataset)

    return epoch_loss, epoch_acc 

if __name__ == '__main__':
    model = load_model(cfg.model_name)
    model = nn.DataParallel(model)
    model.to(device=device)

    criterion = nn.CrossEntropyLoss()

    start = time.time()
    loss,acc=validation(model, test_data_gen, criterion)
    print("Model Test Loss == {}% ".format(loss*100))
    print("Model Accuracy == {}% ".format(acc*100))
    print("Test Time == {}".format(time.time()-start))