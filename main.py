import configparser
import argparse
import logging
import os
import warnings
import torch
from fl import FL
def read_config():
    config = configparser.ConfigParser()
    config.read(r'config\opp\ablation\A30_B30_AB0_label_B_test_A')
    return config

config = read_config()

fl = FL(config)

fl.start()
