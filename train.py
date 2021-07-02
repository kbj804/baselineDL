""" 학습 코드
"""

import os
import random
import argparse
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import numpy as np
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.get_model import get_model

from modules.dataset import CustomDataset
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
from modules.trainer import CustomTrainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory


# DEBUG
DEBUG = False

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yaml')
config = load_yaml(TRAIN_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']
PIN_MEMORY = config['DATALOADER']['pin_memory']

# MODEL
MODEL = config['MODEL']['model_str']
NUM_FEATURES = config['MODEL']['num_features']
INPUT_DIM = config['MODEL']['input_dim']
ENCODING_DIM = config['MODEL']['encoding_dim']
H_DIMS = config['MODEL']['h_dims']

# TRAIN
EPOCHS = config['TRAIN']['num_epochs']
BATCH_SIZE = config['TRAIN']['batch_size']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_fn']
METRIC_FN = config['TRAIN']['metric_fn']

# TRAIN SERIAL
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'{TRAIN_TIMESTAMP}_{MODEL}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']


if __name__ == '__main__':

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set train result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger
    system_logger = get_logger(name='train',
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))

    # Load dataset & dataloader
    train_dataset = CustomDataset(root_dir=PROJECT_DIR, result_dir=PERFORMANCE_RECORD_DIR, config=config, mode='train')
    validation_dataset = CustomDataset(root_dir=PROJECT_DIR, result_dir=PERFORMANCE_RECORD_DIR, config=config, mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

    # Load Model
    Model = get_model(model_str=MODEL)
    model = Model(input_dim=INPUT_DIM, encoding_dim=ENCODING_DIM, h_dims=H_DIMS).to(device)
    system_logger.info('===== Review Model Architecture =====')
    system_logger.info(f'{model} \n')

    # Set optimizer, scheduler, loss function, metric function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(train_dataloader))
    loss_fn = nn.MSELoss()
    metric_fn = mean_squared_error

    # Set trainer
    trainer = CustomTrainer(model, device, loss_fn, metric_fn, optimizer, scheduler, logger=system_logger)

    # Set earlystopper
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=system_logger)

    # Set performance recorder
    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_TIMESTAMP,
        MODEL,
        OPTIMIZER,
        LOSS_FN,
        METRIC_FN,
        EARLY_STOPPING_PATIENCE,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        RANDOM_SEED]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler)

    # Save config yaml file
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config.yaml'), config)

    # Train
    for epoch_index in tqdm(range(EPOCHS)):

        trainer.train_epoch(train_dataloader, epoch_index=epoch_index, verbose=False)
        trainer.validate_epoch(validation_dataloader, epoch_index=epoch_index, verbose=False)

        # Performance record - csv & save elapsed_time
        performance_recorder.add_row(epoch_index=epoch_index,
                                     train_loss=trainer.train_loss_mean,
                                     validation_loss=trainer.validation_loss_mean,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score)

        # Performance record - plot
        performance_recorder.save_performance_plot(final_epoch=epoch_index)

        # early_stopping check
        early_stopper.check_early_stopping(loss=trainer.validation_loss_mean)
        
        if early_stopper.stop:
            break

        trainer.clear_history()

    # last model save
    performance_recorder.weight_path = os.path.join(PERFORMANCE_RECORD_DIR, 'last.pt')
    performance_recorder.save_weight()

