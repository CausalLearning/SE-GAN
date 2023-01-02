import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from model.Networks import DeeplabMulti
from dataset.cityscapes16_dataset import cityscapes16DataSet
import os
from PIL import Image
from utils.tools import *
import torch.backends.cudnn as cudnn

import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():

    parser = argparse.ArgumentParser(description="SEGAN")

    parser.add_argument("--data_dir", type=str, default='/home/yonghao.xu/Data/SegmentationData/Cityscapes/',
                        help="target dataset path.")
    parser.add_argument("--data_list", type=str, default='./dataset/cityscapes_labellist_train.txt',
                        help="target dataset list file.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="the index of the label to ignore in the training.")
    parser.add_argument("--num-classes", type=int, default=16,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='/home/yonghao.xu/PreTrainedModel/Synthia2Cityscapes_batch_24500_miou_450.pth',
                        help="restored model.")   
    parser.add_argument("--snapshot_dir", type=str, default='./PseudoLabel/Synthia/adv1e3/',
                        help="Path to save result.")
    return parser.parse_args()

def flip_tensor(tensor):

    inv_idx = torch.arange(tensor.size(-1)-1,-1,-1).long().cuda()  # N x C x H x W
    inv_tensor = tensor.index_select(-1,inv_idx)

    return inv_tensor

def main():
    
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    f = open(args.snapshot_dir+'GenPseudoLabel_Synthia.txt', 'w')
    
    cudnn.enabled = True
    cudnn.benchmark = True
    model = DeeplabMulti(num_classes=args.num_classes)

    for name, param in model.named_parameters():    
        model.requires_grad=False

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()
    testloader = data.DataLoader(cityscapes16DataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set='train'),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(512,1024), mode='bilinear', align_corners=True)
    interp_scale1 = nn.Upsample(size=(int(512*0.75),int(1024*0.75)), mode='bilinear', align_corners=True)    
    interp_scale2 = nn.Upsample(size=(int(512*1.25),int(1024*1.25)), mode='bilinear', align_corners=True)
    interp_scale3 = nn.Upsample(size=(int(512*1.5),int(1024*1.5)), mode='bilinear', align_corners=True)
    
    length = len(testloader)
   
    for index, batch in enumerate(testloader):
        
        image, _,_, name = batch
        image = image.cuda()
        _,output = model(image)
        output = output.detach()
        output_fusion = F.softmax(interp(output),1)                
        
        _,output = model(flip_tensor(image))
        output = output.detach()
        output_fusion += interp(F.softmax(flip_tensor(output),1))
        
        image_scale1 = interp_scale1(image)
        _,output = model(image_scale1)
        output = output.detach()
        output_fusion += interp(F.softmax(output,1))
        _,output = model(flip_tensor(image_scale1))
        output = output.detach()
        output_fusion += interp(F.softmax(flip_tensor(output),1))
    
        image_scale2 = interp_scale2(image)
        _,output = model(image_scale2)
        output = output.detach()
        output_fusion += interp(F.softmax(output,1))
        _,output = model(flip_tensor(image_scale2))
        output = output.detach()
        output_fusion += interp(F.softmax(flip_tensor(output),1))
        
        image_scale3 = interp_scale3(image)
        h = int(512*1.5/2)
        w = int(1024*1.5/2)
        _,output1 = model(image_scale3[:,:,:h,:w])
        output1 = output1.detach()
        _,output2 = model(image_scale3[:,:,:h,w:])
        output2 = output2.detach()
        _,output3 = model(image_scale3[:,:,h:,:w])
        output3 = output3.detach()
        _,output4 = model(image_scale3[:,:,h:,w:])
        output4 = output4.detach()
        output1 = torch.cat([output1,output2],-1)
        output3 = torch.cat([output3,output4],-1)
        output = torch.cat([output1,output3],-2)
        output_fusion += interp(F.softmax(output.detach(),1))
                
        image_scale3 = flip_tensor(image_scale3)
        _,output1 = model(image_scale3[:,:,:h,:w])
        output1 = output1.detach()
        _,output2 = model(image_scale3[:,:,:h,w:])
        output2 = output2.detach()
        _,output3 = model(image_scale3[:,:,h:,:w])
        output3 = output3.detach()
        _,output4 = model(image_scale3[:,:,h:,w:])
        output4 = output4.detach()
        output1 = torch.cat([output1,output2],-1)
        output3 = torch.cat([output3,output4],-1)
        output = torch.cat([output1,output3],-2)
        output_fusion += interp(F.softmax(flip_tensor(output.detach()),1))
      
        _, predicted = torch.max(output_fusion, 1)
        pred = predicted.squeeze().cpu().numpy()        
          
        output = np.asarray(pred, dtype=np.uint8)       
        name = name[0].split('/')[-1]
        
        output = Image.fromarray(output)
        output.save('%s/%s_PseudoLabel.png' % (args.snapshot_dir, name.split('.')[0]))
        print('{:d} / {:d}:'.format(index, length)+name+'\n')            
        f.write('{:d} / {:d}:'.format(index, length)+name+'\n')
        f.flush() 
    f.close()

if __name__ == '__main__':
    main()
