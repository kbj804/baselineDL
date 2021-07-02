"""Dataset 클래스 정의
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import Dataset

from modules.utils import load_csv


class CustomDataset(Dataset):
    """CustomDataset 클래스 정의
    
    args:
        data_dir (str)              :
        preprocess_serial (str)     :
        mode (str)                  : train, val, test 중 1개
    """

    def __init__(self, root_dir, result_dir, config, mode='train'):

        self.mode = mode
        PREPROCESS_SERIAL = config['PREPROCESS']['preprocess_serial']
        SCALER_STR = config['PREPROCESS']['scaler']
        print(root_dir)
        self.data_dir = os.path.join(root_dir, 'data', PREPROCESS_SERIAL)
        self.result_dir = result_dir

        self.feature_path = os.path.join(self.data_dir, f'{mode}_features.npy')

        print(f'Loading {self.feature_path}...')
        with open(self.feature_path, 'rb') as f:
            features = np.load(f)
        print('Loaded')

        # scaling
        if SCALER_STR is not None:
            if SCALER_STR == 'std':
                feature_scaler = StandardScaler()
            elif SCALER_STR == 'minmax':
                feature_scaler = MinMaxScaler()
            else:
                sys.exit()

            if mode == 'train':
                feature_scaler_path = os.path.join(result_dir, 'feature_scaler.pkl')
                feature_scaler.fit(features)
                joblib.dump(feature_scaler, feature_scaler_path)
            elif mode == 'val':
                feature_scaler_path = os.path.join(result_dir, 'feature_scaler.pkl')
                feature_scaler = joblib.load(feature_scaler_path)
            elif mode == 'test':
                feature_scaler_path = os.path.join(root_dir, 'results', 'train', config['PREDICT']['train_serial'], 'feature_scaler.pkl')
                feature_scaler = joblib.load(feature_scaler_path)
            else:
                sys.exit()

            self.features = feature_scaler.transform(features)

        else:
            self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float)
        return {'feature': feature}

