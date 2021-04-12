# -*- coding: utf-8 -*-
# @Date    : April 11, 2021
# @Author  : XD
# @Blog    ：eadst.com


import datetime


class Configs():

    # 1.string parameters
    dataset_name = 'AID'
    model_name = 'ALL'
    all_data = './data/{}/'.format(dataset_name)
    train_data = './data/train_{}/'.format(dataset_name)
    test_data = './data/test_{}/'.format(dataset_name)
    eval_data = './data/eval_{}/'.format(dataset_name)
    checkpoint = './checkpoints/{}_ALL/'.format(dataset_name)
    best_models = checkpoint + 'best_model/'
    time_str = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    logs = './logs/ALL_{}_{}.log'.format(dataset_name, time_str)
    gpus = '0'
    pretrained = True
    resume = False

    # 2.numeric parameters
    epochs = 50
    batch_size = 16
    seed = 888
    # CRLUC's lr and weight decay
    lr = 1e-4
    weight_decay = 1e-5
    # the other models' lr and weight decay
    lr_2 = 1e-3
    weight_decay_2 = 1e-6
    step_size = 20
    gamma = 0.1
    start_epoch = 1

    lstm_out_planes = 32
    fc_input_num = 2048  # 4096
    num_classes = 30


config = Configs()
