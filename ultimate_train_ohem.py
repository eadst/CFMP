# -*- coding: utf-8 -*-
# @Date    : April 11, 2021
# @Author  : XD
# @Blog    ：eadst.com


import os
import time
import random
import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from torch.optim import Adam
import os
from utils.log_settings import Logger
from dataset.dataloader import *
import importlib
from models.model_loader import *


config_name = 'aid_config'
config_file = importlib.import_module('config.' + config_name)
config = config_file.config
# Define training parameters
dataset_name = config.dataset_name
resume = config.resume
train_data_root = config.train_data
test_data_root = config.test_data
model_name = config.model_name
output_dir = config.checkpoint
open_ohem = True

# Initialize random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# Set training device
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

# Create log file
logging = Logger(config.logs)
logging.info(device)
logging.info('dataset name: ' + dataset_name)
logging.info('model name: ' + model_name)
logging.info('config name: ' + config_name)


# Load dataset
def load_data(split_rate=0.6):
    logging.info('split_rate: ' + str(split_rate))
    total_dataset = datasets.ImageFolder(config.all_data)
    train_size = int(split_rate * len(total_dataset))
    val_size = int((len(total_dataset) - train_size)/2)
    test_size = len(total_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(total_dataset, [train_size, val_size, test_size])

    train_dataset = PreprocessData(train_dataset, dataset_name, format='ultimate')
    train_dataset_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = PreprocessData(val_dataset, dataset_name, mode='val', format='ultimate')
    val_dataset_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = PreprocessData(test_dataset, dataset_name, mode='test', format='ultimate')
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    return train_dataset_loader, val_dataset_loader, test_dataset_loader


def val(val_dataset_loader, model, criterion):
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    val_loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataset_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            step_loss = val_loss / (i + 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accuracy = 100.0 * (correct / total)
    logging.info('val_loss: ' + str(val_loss))
    logging.info('val_step_loss: ' + str(step_loss))
    logging.info('val_accuracy: ' + str(accuracy))
    return accuracy


def true_table(dataset_loader, model):
    num_class = config.num_classes
    count_mat = np.zeros((num_class, num_class))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataset_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for batch_i in range(len(predicted)):
                count_mat[int(labels[batch_i])-1][predicted[batch_i]] += 1

    result_mat = np.zeros((num_class, num_class))
    for row, lst in enumerate(count_mat):
        total_lst = sum(lst)
        s = ''
        for col, e in enumerate(lst):
            rate = round(100.0 * e / float(total_lst), 2)
            result_mat[row][col] = rate
            s += str(rate) + ', '
        logging.info('class_result {}: '.format(row+1) + s[:-2])


def test(test_dataset_loader, model, criterion):
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataset_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            step_loss = test_loss / (i + 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accuracy = 100.0 * (correct / total)
    logging.info('test_loss: ' + str(test_loss))
    logging.info('test_step_loss: ' + str(step_loss))
    logging.info('test_accuracy: ' + str(accuracy))
    return accuracy


# Train model
def train(model_name, split_rate, iter, train_dataset_loader, val_dataset_loader, test_dataset_loader, open_ohem):
    # Set model and optimizer
    model = load_model(model_name, config.num_classes, use_pretrained=True)
    lr = config.lr
    weight_decay = config.weight_decay
    # model = None
    if model:
        logging.info('model name: ' + model_name)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        best_train_accuracy = 0.0
        best_val_accuracy = 0.0
        best_test_accuracy = 0.0
        best_model_info = [model, best_train_accuracy, best_val_accuracy]
        # Start training
        model.train()
        for epoch in range(config.start_epoch, config.epochs+1):
            model.train()
            scheduler.step(epoch)
            logging.info('Iteration ' + str(iter) + ' Epoch ' + str(epoch) + ' training...')
            # Set parameters
            train_loss = 0.0
            step_loss = 0.0
            accuracy = 0.0
            correct = 0
            total = 0
            # ohem init
            ohem = []

            for i, (inputs, labels) in enumerate(train_dataset_loader):
                # print(inputs.size(0), labels.size(0))
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                train_loss += loss.item()
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                step_loss = train_loss / (i + 1)
                accuracy = 100.0 * (correct / total)
                if correct < total and epoch > 20 and open_ohem:
                    ohem.append([inputs, labels])
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logging.info('train_loss: ' + str(train_loss))
            logging.info('step_loss: ' + str(step_loss))
            logging.info('accuracy: ' + str(accuracy))
            if ohem:
                print("This part is using ohem training...")
                train_loss = 0.0
                correct = 0
                total = 0
                for i, (inputs, labels) in enumerate(ohem):
                    outputs = model(inputs)
                    # Loss calculate
                    loss = criterion(outputs, labels)
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    step_loss = train_loss / (i + 1)
                    accuracy = 100.0 * (correct / total)
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                logging.info('ohem_train_loss: ' + str(train_loss))
                logging.info('ohem_step_loss: ' + str(step_loss))
                logging.info('ohem_accuracy: ' + str(accuracy))
            if accuracy > best_train_accuracy:
                state = {
                    'net': model.state_dict(),
                    'accuracy': accuracy,
                    'epoch': epoch
                }
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                net_save_path = '{}train_{}_{}_{}_{}.pth'.format(output_dir, str(split_rate), str(iter), dataset_name, model_name)
                torch.save(state, net_save_path)
                best_train_accuracy = accuracy
                logging.info('train accuracy > best_train_accuracy, saving net')

            logging.info('Iteration ' + str(iter) + ' Epoch ' + str(epoch) + ' validating...')
            val_accuracy = val(val_dataset_loader, model, criterion)

            if val_accuracy > best_val_accuracy:
                state = {
                    'net': model.state_dict(),
                    'accuracy': val_accuracy,
                    'epoch': epoch
                }
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                net_save_path = '{}val_{}_{}_{}_{}.pth'.format(output_dir, str(split_rate), str(iter), dataset_name, model_name)
                torch.save(state, net_save_path)
                best_val_accuracy = val_accuracy
                logging.info('val_accuracy > best_val_accuracy, saving net')
                best_model_info = [model, best_train_accuracy, best_val_accuracy]
            print('|O(∩_∩)O| {} current best train accuracy: '.format(model_name), best_train_accuracy)
            print('|O(∩_∩)O| {} current best val accuracy: '.format(model_name), best_val_accuracy)

        logging.info('Iteration ' + str(iter) + ' testing...')
        model = best_model_info[0]
        test_accuracy = test(test_dataset_loader, model, criterion)
        state = {
            'net': model.state_dict(),
            'accuracy': test_accuracy
        }
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        net_save_path = '{}test_{}_{}_{}_{}.pth'.format(output_dir, str(split_rate), str(iter), dataset_name, model_name)
        torch.save(state, net_save_path)
        best_train_accuracy, best_val_accuracy, best_test_accuracy = best_model_info[1], best_model_info[2], test_accuracy
        logging.info('saving test net')

        logging.info('train true table')
        true_table(train_dataset_loader, model)
        logging.info('val true table')
        true_table(val_dataset_loader, model)
        logging.info('test true table')
        true_table(test_dataset_loader, model)
        logging.info('{} best train accuracy: '.format(model_name) + str(best_train_accuracy))
        logging.info('{} best val accuracy: '.format(model_name) + str(best_val_accuracy))
        logging.info('{} best test accuracy: '.format(model_name) + str(best_test_accuracy))
        return best_train_accuracy, best_val_accuracy, best_test_accuracy
    else:
        return 0, 0, 0


if __name__ == '__main__':
    # The first 'resnet50-0' using OHEM, the second 'resnet50-0' without OHEM.
    model_list = ['resnet50-0', 'resnet50-0']
    iterations = 10
    split_rate = 0.6
    accuracy_list = []
    record = []
    for idx in range(len(model_list)):
        accuracy_list.append([0.0, 0.0, 0.0])
    # train_dataset_loader, val_dataset_loader, test_dataset_loader = load_data(split_rate)
    for iter in range(iterations):
        train_dataset_loader, val_dataset_loader, test_dataset_loader = load_data(split_rate)
        for m_idx, model_name in enumerate(model_list):
            if m_idx % 2:
                open_ohem = False
                print("This iter is not using ohem training...")
            else:
                open_ohem = True
                print("This iter is using ohem training...")
            train_accuracy, val_accuracy, test_accuracy = train(model_name, split_rate, iter,
                                                                train_dataset_loader,
                                                                val_dataset_loader,
                                                                test_dataset_loader, open_ohem)
            accuracy_list[m_idx][0] += train_accuracy
            accuracy_list[m_idx][1] += val_accuracy
            accuracy_list[m_idx][2] += test_accuracy
            record.append([model_name, train_accuracy, val_accuracy, test_accuracy])
    print('========== ')
    for accuracy in accuracy_list:
        print(accuracy[0]/iterations, accuracy[1]/iterations, accuracy[2]/iterations)
    print(record)