import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob
import json 
import pandas as pd
from pathlib import Path
import PIL
import cv2 
import tqdm
from multiprocessing import Pool
import argparse

# set number of CPUs to run on
ncore = "12"
# set env variables
# have to set these before importing numpy
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore


my_parser = argparse.ArgumentParser()
my_parser.add_argument('--genNum', type=int, help='Number from genframe')
my_parser.add_argument('--data_root', type=str, default='/media/SSD/Frame_Inter_rheology2023/_10GenFrame/FLAVRModel/Frame_Inter/Saliva2', help='path to text files')
args = my_parser.parse_args()

genNum = args.genNum
genNum = f'gen{genNum}'
data_root = args.data_root
data_root = f'{data_root}/{genNum}/'
print(f'Data Set : [ {data_root}]')
print(f'-'*100)

    
##function check broken images path.
def img_verify(sub_testlist):
    _except = []
    for file in sub_testlist:
        try:
            img = Image.open(file)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            _except.append(file)
    return _except 

def Filter_dataset(data_root): ## subset == train or test -->> type must be string
    ### Start
    data_fn = glob.glob(f"{data_root}*.txt")
    data_fn.sort()
    ## For training set
    for fn in data_fn:
        trainlist0 = []
        NameFn = fn
        with open(fn, 'r') as txt:
             meta_data = [line.strip() for line in txt]
        for seq in meta_data:
            img1_path, img2_path, img3_path, img4_path = seq.split(' ')
            trainlist0.append([img1_path,img2_path,img3_path,img4_path])
            #time.sleep(0.01)
        print(f"Data set [File Name]: =============== {NameFn} ===============")
        print(f"**Data set** with Data size: {len(trainlist0)} batch")
        time.sleep(0.05)     
        ###''' 
        pth_training1, pth_training2, pth_training3, pth_training4 = [],[],[],[]
        root2save = NameFn.replace("/media/SSD/Frame_Inter_rheology2023/_10GenFrame/FLAVRModel/Frame_Inter/", "/media/SSD/Frame_Inter_rheology2023/_10GenFrame/FLAVRModel/broken-images/pred_text/")
        rootdir = root2save.split('/')[:-1]
        rootdir_ = '/'.join(rootdir)
        if not os.path.exists(rootdir_):
            os.makedirs(rootdir_)
        for index in range(len(trainlist0)):
            print(f"Load images : {index+1}")
            imgpaths = trainlist0[index]
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            # Load images
            _except = img_verify(imgpaths)
            if len(_except) == 0:
                pth_training1.append(imgpaths[0])
                pth_training2.append(imgpaths[1])
                pth_training3.append(imgpaths[2])
                pth_training4.append(imgpaths[3])
            #time.sleep(0.02)
        time.sleep(0.09)
        print("========== Create DataFrame ========== ")
        ## Prepare to create text files
        df_ = pd.DataFrame(
            {'Path1': pth_training1,
             'Path2': pth_training2,
             'Path3': pth_training3,
             'Path4': pth_training4,
            })
        print("========== Prepare to create text files ========== ")
        time.sleep(0.5)
        df_['Path_txt'] = ''
        for i in range(len(df_)):
            name1 = df_['Path1'][i]
            name2 = df_['Path2'][i]
            name3 = df_['Path3'][i]
            name4 = df_['Path4'][i]
            df_.loc[df_.index[i], 'Path_txt'] = str(name1)+' '+str(name2)+' '+str(name3)+' '+str(name4)    
        print(f'Filtered Data set with shape : {df_.shape}')
        ## Save to text file
        #time.sleep(0.5)
        print("On process to Save text file")
        list_path = df_['Path_txt'].tolist()
        with open(root2save, 'w') as f:
             for line in list_path:
                f.write(f"{line}\n")
        print(f'Done!! : Write text file name -> [ {root2save} ] ')

## Run Function 
start = time.time()
Filter_dataset(data_root)
end = time.time()
print(f'Filter Data set, That took {round(end-start, 3)} seconds')
         
