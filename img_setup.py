import os

data_path = 'data'

def setup_csv(name):
    """Reads images into csv"""
    with open(os.path.join(data_path, f'{name}.csv'), 'w') as f:
        data_list = []
        
        for fname in os.listdir(os.path.join(data_path, f'{name}/p')):
            data_list.append(f"{name}/p/{fname},1")
        for fname in os.listdir(os.path.join(data_path, f'{name}/n')):
            data_list.append(f"{name}/n/{fname},0")

        f.write('\n'.join(data_list))


setup_csv('train')
setup_csv('test')
