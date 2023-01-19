import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob


class Middelburry(Dataset):
    #ef __init__(self, data_root , ext="png"):
    #def __init__(self, data_root , ext="jpg"):
    def __init__(self, data, ext="jpg"):

        super().__init__()

        #self.data_root = data_root
        self.data = data
        
        self.meta_data = []
#         test_fn = glob.glob(f"{self.data_root}*-4linedemo.txt")
#         test_fn = test_fn[0] 
        #with open(self.data_root, 'r') as txt:
        with open(self.data, 'r') as txt:
             testlist = [line.strip() for line in txt]
        for seq in testlist:
            img1_path, img2_path, img3_path, img4_path = seq.split(' ')
            self.meta_data.append([img1_path,img2_path,img3_path,img4_path])

            
        self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):

#         imgpath = os.path.join(self.data_root , self.file_list[idx])
#         name = self.file_list[idx]
#         if name == "Teddy": ## Handle inputs with just two inout frames. FLAVR takes atleast 4.
#             imgpaths = [os.path.join(imgpath , "frame10.png") , os.path.join(imgpath , "frame10.png") ,os.path.join(imgpath , "frame11.png") ,os.path.join(imgpath , "frame11.png") ]
#         else:
#             imgpaths = [os.path.join(imgpath , "frame09.png") , os.path.join(imgpath , "frame10.png") ,os.path.join(imgpath , "frame11.png") ,os.path.join(imgpath , "frame12.png") ]

        imgpaths = self.meta_data[idx]
        #name = self.meta_data[idx]
        images = [Image.open(img).convert('RGB') for img in imgpaths]
        images = [self.transforms(img) for img in images]

        sizes = images[0].shape
        
        return images , imgpaths 
    

    def __len__(self):

        #return len(self.file_list)
        return len(self.meta_data)

#def get_loader(data_root, batch_size, shuffle, num_workers, test_mode=True):
def get_loader(data, batch_size, shuffle, num_workers, test_mode=True):

    #dataset =  Middelburry(data_root)
    dataset =  Middelburry(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
