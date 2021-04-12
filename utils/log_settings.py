# -*- coding: utf-8 -*-
# @Date    : April 11, 2021
# @Author  : XD
# @Blog    ：eadst.com


import logging


def Logger(filename='./logs/default.log'):
    logging.basicConfig(
                        level=logging.DEBUG,
                        format='%(asctime)s-%(levelname)s-%(message)s',
                        datefmt='%y-%m-%d %H:%M',
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging