import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob
import pandas as pd

##function check broken images path.
# def img_verify(sub_testlist):
#     _except = []
#     for file in sub_testlist:
#         try:
#             img = Image.open(file)  # open the image file
#             img.verify()  # verify that it is, in fact an image
#         except (IOError, SyntaxError) as e:
#             _except.append(file)
#     return _except 

class VimeoSepTuplet(Dataset):
    def __init__(self, data_root, is_training , input_frames="1357"):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = data_root
        #self.image_root = os.path.join(self.data_root, 'sequences')
        self.trainlist = [] 
        self.testlist = [] 
        self.training = is_training
        self.inputs = input_frames

        #train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')  ## Chang path if change fold. 
#         train_fn = glob.glob(f"{data_root}/*train.txt")
#         train_fn.sort()
        train_fn = pd.read_csv(f"{data_root}/GlycerolRheology2023-train.csv") ## Train Only Glycerol
        #test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        test_fn = glob.glob(f"{data_root}/rheology2023-test.txt")
        test_fn.sort()
        ## For training set
#         for fn in train_fn:
#             with open(fn, 'r') as txt:
#                  meta_data = [line.strip() for line in txt]
#             for seq in meta_data:
#                 img1_path, img2_path, img3_path, img4_path, img5_path = seq.split(' ')
#                 self.trainlist.append([img1_path,img2_path,img3_path,img4_path, img5_path])
        for i in range(len(train_fn)):
            img1_path = train_fn['Path1'][i]
            img2_path = train_fn['Path2'][i]
            img3_path = train_fn['Path3'][i]
            img4_path = train_fn['Path4'][i]
            img5_path = train_fn['Path5'][i]
            self.trainlist.append([img1_path,img2_path,img3_path,img4_path, img5_path])
       ## For Test set 
        for ft in test_fn:
            with open(ft, 'r') as txt:
                 meta_data = [line.strip() for line in txt]
            for seq in meta_data:
                img1_path, img2_path, img3_path, img4_path, img5_path = seq.split(' ')
                self.testlist.append([img1_path,img2_path,img3_path,img4_path, img5_path])
    

        if self.training:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
              transforms.ToTensor()
             ])
   

    def __getitem__(self, index):
        if self.training:
            imgpaths = self.trainlist[index]
        else:
            imgpaths = self.testlist[index]
        
        pth_ = imgpaths
        
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Load images
        images = [Image.open(pth) for pth in imgpaths]

        ## Select only relevant inputs
#         inputs = [int(e)-1 for e in list(self.inputs)]
#         inputs = inputs[:len(inputs)//2] + [3] + inputs[len(inputs)//2:]
#         images = [images[i] for i in inputs]
#         imgpaths = [imgpaths[i] for i in inputs]
        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
            images_ = []
            for img_ in images:
                random.seed(seed)
                images_.append(self.transforms(img_))
            images = images_
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]
        else:
            T = self.transforms
            images = [T(img_) for img_ in images]

        gt = images[len(images)//2]
        images = images[:len(images)//2] + images[len(images)//2+1:]
        
        return images, [gt]

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)

def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoSepTuplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":

    #dataset = VimeoSepTuplet("./vimeo_septuplet/", is_training=True)
    dataset = VimeoSepTuplet("/media/SSD/Frame_Inter_rheology2023/dataset/_5Frame/", is_training=True)  ## Path root
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=32, pin_memory=True)