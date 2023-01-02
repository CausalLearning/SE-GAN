import argparse
import torch
from torch.utils import data
import numpy as np
import os
from utils.tools import *
from dataset.gta5_dataset import GTA5DataSet
from model.Networks import TransformerNet
from tqdm import tqdm
from PIL import Image
import torch.backends.cudnn as cudnn

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():

    parser = argparse.ArgumentParser(description="SEGAN")
    
    #dataset
    parser.add_argument("--data_dir_source", type=str, default='/home/yonghao.xu/Data/SegmentationData/GTA5/',
                        help="source dataset path.")
    parser.add_argument("--data_list_source", type=str, default='./dataset/GTA5_imagelist_train.txt',
                        help="source dataset list file.")
    parser.add_argument("--input_size", type=str, default='1024,512',
                        help="width and height of input images.")    
    parser.add_argument("--original_size", type=str, default='1914,1052',
                        help="width and height of original images.")          

    #network
    parser.add_argument("--batch_size", type=int, default=4,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--restore_from", type=str, default='/home/yonghao.xu/PreTrainedModel/GTATrans_epoch_5_batch_12482_miou_50.pth',
                        help="pretrained Seg model.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./TGSTN/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

def main():

    args = get_arguments()
    
    snapshot_dir = args.snapshot_dir+'GTA/images/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.original_size.split(','))
    original_size = (w, h)

    # Create network
    trans_net = TransformerNet()   

    trans_net.load_state_dict(torch.load(args.restore_from))
    for name, param in trans_net.named_parameters():
        param.requires_grad=False
    trans_net = trans_net.cuda()
   
    src_loader = data.DataLoader(
                    GTA5DataSet(args.data_dir_source, args.data_list_source,
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    tbar = tqdm(src_loader)
    trans_net.eval()
    cudnn.enabled = True
    cudnn.benchmark = True

    for batch_index, src_data in enumerate(tbar):
        images, _, _, im_name = src_data
        images = images.cuda()
               
        trans_imgs = trans_net(images).cpu()
        for i in range(len(im_name)):
            name = im_name[i]         

            im = trans_imgs[i].data.numpy().copy().transpose(1, 2, 0)
            im += IMG_MEAN
            im = im[:, :, ::-1] 
            im[im > 255] = 255
            im[im < 0] = 0
            im = Image.fromarray(np.uint8(im))
            im = im.resize(original_size, Image.BICUBIC)       
            im.save(snapshot_dir+name,'png')  

if __name__ == '__main__':
    main()
