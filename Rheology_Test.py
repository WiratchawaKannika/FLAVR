import os
import sys
import time
import copy
import shutil
import random
import pdb
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import glob
from torch.utils.tensorboard import SummaryWriter

from dataset.transforms import Resize

import config
import myutils

from torch.utils.data import DataLoader
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]='1'
args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)
             
##Get dataset
#from dataset.Middleburry import get_loader
#test_loader = get_loader(args.data_root, 1, shuffle=False, num_workers=args.num_workers)   

from model.FLAVR_arch import UNet_3D_3D
print("Building model: %s"%args.model.lower())
model = UNet_3D_3D(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType)

# Just make every model to DataParallel
model = torch.nn.DataParallel(model).to(device)
print("#params" , sum([p.numel() for p in model.parameters()]))

def make_image(img):
    # img = F.interpolate(img.unsqueeze(0) , (720,1280) , mode="bilinear").squeeze(0)
    q_im = img.data.mul(255.).clamp(0,255).round()
    im = q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return im

##Get dataset ***--Setting--*** 
from dataset.Middleburry import get_loader
#data_frame = '/home/kannika/codes_AI/CSV/rheology2023_random40folder_4linedemo.csv' ## ** dataset for predict. 
# for data_root in data_path:
#     test_loader = get_loader(data_root, 1, shuffle=False, num_workers=args.num_workers)  
#test_loader = get_loader(args.data_root, 1, shuffle=False, num_workers=args.num_workers)  
def test(args):
    time_taken = []
    img_save_id = 0
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()

    psnr_list = []
    with torch.no_grad():
        if args.GenBroken:
            data = args.data_root
            test_loader = get_loader(data, 1, shuffle=False, num_workers=args.num_workers)  
            for i, (images, name ) in enumerate((test_loader)): ##** Modified by AI
                    name2 = name[1] ## File name from Frame 2
                    name2 = name2[0]
                    ### Create name img path
                    name2_ = name2.split('/')[-1]
                    folder_name = name2.split('/')[:-1]
                    folder_name_ = '/'.join(folder_name)
                    save_pathimg = folder_name_.replace("rheology2023", "Frame_Inter_rheology2023/broken-images/Frame_Inter/FLAVRModel")
                    import imageio
                    os.makedirs(save_pathimg, exist_ok=True) 
                    
                    images = torch.stack(images , dim=1).squeeze(0)
                    # images = [img_.cuda() for img_ in images]
                    H,W = images[0].shape[-2:]
                    resizes = 8*(H//8) , 8*(W//8)

                    import torchvision
                    transform = Resize(resizes)
                    rev_transforms = Resize((H,W))
                    images = transform(images).unsqueeze(0).cuda()# [transform(img_.squeeze(0)).unsqueeze(0).cuda() for img_ in images]
                    images = torch.unbind(images, dim=1)

                    start_time = time.time()
                    out = model(images)
                    print("Time Taken" , time.time() - start_time)

                    out = torch.cat(out)
                    out = rev_transforms(out)

                    output_image = make_image(out.squeeze(0))
                    
                    out_name = os.path.join(save_pathimg, os.path.basename(name2_).split('.')[0]+'_inter'+'.png')
    #                 import imageio
    #                 os.makedirs(save_path, exist_ok=True) ## set to True, won't get a FileExistsError if the target directory already exists.
                    imageio.imwrite(out_name, output_image) 
            SHOW_pathimg = save_pathimg.split('/')[:7]
            SHOW_pathimg_ = '/'.join(SHOW_pathimg) 
            print('Frame Interpolation saVe at -->>', SHOW_pathimg_)
            print('*'*120)
        else:
            genNum = args.genNum
            _genNum = f'gen{genNum}'
            genNum_old = genNum-1
            _genNumold = f'gen{genNum_old}'
            data_root = args.data_root
            data_path = glob.glob(f"{data_root}/*-4linedemo.txt")
            data_path.sort()
            for data in data_path:  ##** Modified by AI
                print(f'On Process Folder  -->> [ {data} ]')
                list_imgframe = list()
                test_loader = get_loader(data, 1, shuffle=False, num_workers=args.num_workers)   ##** Modified by AI
                if genNum == 1:
                    folder_name_ = data.replace("-4linedemo", f"_{_genNum}-inter")
                    folder_name_ = folder_name_.split('.')[0]
                    save_pathimg = folder_name_.replace("pred_text/Saliva2/origin", f"Frame_Inter/Saliva2/{_genNum}")
                else:
                    folder_name_ = data.replace(f"{_genNumold}-4linedemo", f"{_genNum}-inter")
                    folder_name_ = folder_name_.split('.')[0]
                    save_pathimg = folder_name_.replace(_genNumold, _genNum)
                ##**Mkdir Directory 
                import imageio
                os.makedirs(save_pathimg, exist_ok=True)
                ### Create name img path
                name_img = save_pathimg.split("/")[-1]
                name_img_ = name_img.split("_")[:-1]
                __name_img = '_'.join(name_img_)
                ## Create path to save CSV.
                save_csv = save_pathimg.split("/")[:-1]
                save_csv_ = '/'.join(save_csv)
                pathName_csv = save_csv_+'/'+__name_img+'_'+_genNum+'.csv'
                _pathName_csv = pathName_csv.replace(_genNumold, f"Frame_Inter/Saliva2/{_genNum}")
                
                for i, (images, name ) in enumerate((test_loader)): ##** Modified by AI
                    images = torch.stack(images , dim=1).squeeze(0)
                    # images = [img_.cuda() for img_ in images]
                    H,W = images[0].shape[-2:]
                    resizes = 8*(H//8) , 8*(W//8)

                    import torchvision
                    transform = Resize(resizes)
                    rev_transforms = Resize((H,W))
                    images = transform(images).unsqueeze(0).cuda()# [transform(img_.squeeze(0)).unsqueeze(0).cuda() for img_ in images]
                    images = torch.unbind(images, dim=1)

                    start_time = time.time()
                    out = model(images)
                    print("Time Taken" , time.time() - start_time)

                    out = torch.cat(out)
                    out = rev_transforms(out)

                    output_image = make_image(out.squeeze(0))
                    ## save frame inter
                    out_name = os.path.join(save_pathimg, __name_img+'_inter'+str(i+1)+'_'+_genNum+'.jpg') 
                    imageio.imwrite(out_name, output_image) 
                    ## Save Sequence frame to csv.
                    if i == 0:
                        list_imgframe.append(name[0][0])
                        list_imgframe.append(name[1][0])
                        list_imgframe.append(out_name)
                    elif i == len(test_loader)-1:
                        list_imgframe.append(name[1][0])
                        list_imgframe.append(out_name)
                        list_imgframe.append(name[2][0])
                        list_imgframe.append(name[3][0])
                    else:
                        list_imgframe.append(name[1][0])
                        list_imgframe.append(out_name)
                df = pd.DataFrame(list_imgframe, columns =['seq_inter'])
                df.to_csv(_pathName_csv)
                print('Frame Interpolation saVe at -->>', save_pathimg)
                print(f"Save Sequence Dataframe at -->> {_pathName_csv} With Shape: {df.shape}")
                print('*'*120)

    return

def main(args):
    
    assert args.load_from is not None

    model_dict = model.state_dict()
    model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
    test(args)


if __name__ == "__main__":
    main(args)
