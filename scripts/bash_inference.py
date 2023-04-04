import os
import numpy as np
from glob import glob
import subprocess
from tqdm import tqdm

if __name__=='__main__':
    # image_list = sorted(glob(os.path.join('/SSDe/kisane_DB/kisane_DB_v0_3/multi_data/*/*/*/*/*/*_Color.png')))
    image_list = sorted(glob(os.path.join('/SSDe/kisane_DB/kisane_DB_v0_3/multi_data/G04-1014/TRAY2/DR/TP2/TO135/*_Color.png')))
    for image_path in image_list:
        file_name = os.path.basename(image_path)
        save_path = 'visualization/%s' %file_name
        
        cmd = 'python inference.py --data_path %s --save_path %s --threshold %f' %(image_path, save_path, 0.4)
        subprocess.call(cmd, shell=True)