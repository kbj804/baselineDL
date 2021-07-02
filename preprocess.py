import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import datatable as dt

datasets = ['Training', 'Test']
dataset_names = ['train', 'test']

target_dict = {
    '정상': 0,
    '이상': 1
}


def preprocess():
    for i, dataset in enumerate(datasets):
        print(f'===== {dataset_names[i].upper()} =====')

        base_dir = '.\\Scripts\\autoencoder_baseline\\'
        filename_list_path = os.path.join(base_dir, f'{dataset_names[i]}_filename_list.csv')
        filename_list_df = dt.fread(filename_list_path).to_pandas()
        print(f'{dataset_names[i]} dataset Shape: {filename_list_df.shape}')
        filename_list_arr = np.array(filename_list_df)

        data_li = []
        for row in tqdm(filename_list_arr):
            # print(f'row  = {row}')
            csv_path = base_dir + f"data/{dataset_names[i]}/" +os.path.join(row[0])
            signal_df = dt.fread(csv_path).to_pandas()

            data = signal_df['x'].tolist() + signal_df['y'].tolist() + signal_df['z'].tolist()
            data_li.append(data)

        data_arr = np.array(data_li)
        data_npy_path = os.path.join(base_dir, f'{dataset_names[i]}_features.npy')
        with open(data_npy_path, 'wb') as f:
            np.save(f, data_arr)


if __name__ == '__main__':
    # preprocess()
    # file_list = os.listdir('.\\Scripts\\autoencoder_baseline\\data\\train')
    # df = pd.DataFrame(file_list, columns=['path'])
    # df.to_csv('train_filename_list.csv')
    