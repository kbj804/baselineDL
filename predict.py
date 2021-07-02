""" 추론 코드
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from sklearn.metrics import mean_squared_error

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from model.get_model import get_model

from modules.dataset import CustomDataset
from modules.recorders import PerformanceRecorder
from modules.trainer import CustomTrainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory


# DEBUG
DEBUG = True

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PREDICT_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/predict_config.yaml')
config = load_yaml(PREDICT_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# PREDICT
TRAIN_SERIAL = config['PREDICT']['train_serial']
BATCH_SIZE = config['PREDICT']['batch_size']

# MODEL CONFIGURATION
TRAIN_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)
TRAIN_CONFIG_PATH = os.path.join(TRAIN_RECORD_DIR, 'train_config.yaml')
train_config = load_yaml(TRAIN_CONFIG_PATH)
MODEL = train_config['MODEL']['model_str']
NUM_FEATURES = train_config['MODEL']['num_features']

# TRAIN CONFIGURATION
EPOCHS = train_config['TRAIN']['num_epochs']
LEARNING_RATE = train_config['TRAIN']['learning_rate']
WEIGHT_DECAY = train_config['TRAIN']['weight_decay']

# PREDICT SERIAL
KST = timezone(timedelta(hours=9))
PREDICT_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
PREDICT_SERIAL = f'{TRAIN_SERIAL.split("_")[0]}_{PREDICT_TIMESTAMP}_{MODEL}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'predict', PREDICT_SERIAL)


if __name__ == '__main__':

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set predict result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger
    system_logger = get_logger(name='predict',
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'predict_log.log'))

    # Load dataset & dataloader
    test_dataset = CustomDataset(root_dir=PROJECT_DIR, result_dir=TRAIN_RECORD_DIR, config=config, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Load Model & its trained weight
    Model = get_model(model_str=MODEL)
    model = Model(NUM_FEATURES).to(device)

    # Set optimizer, scheduler, loss function, metric function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(test_dataloader))
    loss_fn = nn.MSELoss()
    metric_fn = mean_squared_error

    # Set trainer
    trainer = CustomTrainer(model, device, loss_fn, metric_fn, optimizer, scheduler)

    # Save config yaml file
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'predict_config.yaml'), config)

    # Predict
    trainer.predict_epoch(dataloader=test_dataloader, epoch_index=0, verbose=False)

    # Save prediction
    SUBMISSION_PATH = os.path.join(PERFORMANCE_RECORD_DIR, 'submission.csv')

    sub_df = pd.DataFrame(trainer.prediction_score_list)
    sub_df.reset_index(level=0, inplace=True)
    sub_df['ID'] = sub_df['index']
    sub_df['anomaly_score'] = sub_df[0]
    sub_df[['ID', 'anomaly_score']].to_csv(SUBMISSION_PATH, index=False)

