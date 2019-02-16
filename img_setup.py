"""Reads images into csv"""

import os

data_path = 'data'

with open(os.path.join(data_path, 'train.csv'), 'w') as f:
    data_list = []
    
    for fname in os.listdir(os.path.join(data_path, 'imgs/p')):
        data_list.append(f"imgs/p/{fname},1")
    for fname in os.listdir(os.path.join(data_path, 'imgs/n')):
        data_list.append(f"imgs/n/{fname},0")

    f.write('\n'.join(data_list))
