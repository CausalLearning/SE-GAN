import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from model.Networks import DeeplabMulti
from dataset.cityscapes16_dataset import cityscapes16DataSet
import os
from utils.tools import *
import torch.backends.cudnn as cudnn

import torch.nn as nn

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():

    parser = argparse.ArgumentParser(description="SEGAN")

    parser.add_argument("--data_dir", type=str, default='/home/yonghao.xu/Data/SegmentationData/Cityscapes/',
                        help="target dataset path.")
    parser.add_argument("--data_list", type=str, default='./dataset/cityscapes_labellist_val.txt',
                        help="target dataset list file.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="the index of the label to ignore in the training.")
    parser.add_argument("--num-classes", type=int, default=16,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='/home/yonghao.xu/PreTrainedModel/Synthia2Cityscapes_batch_8500_miou_474.pth',
                        help="restored model.")   
    parser.add_argument("--snapshot_dir", type=str, default='./Result/Synthia/',
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
    f = open(args.snapshot_dir+'MultiScaleTest_Synthia.txt', 'w')
    
    cudnn.enabled = True
    cudnn.benchmark = True
    model = DeeplabMulti(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()
    testloader = data.DataLoader(cityscapes16DataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(512,1024), mode='bilinear', align_corners=True)
    interp_scale1 = nn.Upsample(size=(int(512*0.75),int(1024*0.75)), mode='bilinear', align_corners=True)   
    interp_scale2 = nn.Upsample(size=(int(512*1.25),int(1024*1.25)), mode='bilinear', align_corners=True)
    interp_scale3 = nn.Upsample(size=(int(512*1.5),int(1024*1.5)), mode='bilinear', align_corners=True)
    length = len(testloader)

    interp_target = nn.Upsample(size=(1024,2048), mode='bilinear', align_corners=True)
    num_classes = args.num_classes
    hist = np.zeros((num_classes, num_classes))    
    with open('./dataset/info16.json','r') as fp:
        info = json.load(fp)
    name_classes = np.array(info['label'], dtype=np.str)
    for index, batch in enumerate(testloader):  
        image, label,_, name = batch
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
        
        output_fusion = interp_target(output_fusion)
        _, predicted = torch.max(output_fusion, 1)

        pred = predicted.squeeze().cpu().numpy()
        label = label.numpy().squeeze()

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:d}'.format(len(label.flatten()), len(pred.flatten()), index))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if index > 0 and index % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(index, length, 100*np.mean(per_class_iu(hist))))
            f.write('{:d} / {:d}: {:0.2f}\n'.format(index, length, 100*np.mean(per_class_iu(hist)))) 
    
    mIoUs = per_class_iu(hist)
    for index_class in range(num_classes):
        f.write('\n===>' + name_classes[index_class] + ':\t' + str(round(mIoUs[index_class] * 100, 2))) 
        print('===>' + name_classes[index_class] + ':\t' + str(round(mIoUs[index_class] * 100, 2)))
    f.write('\n ===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))+'\n') 
    f.flush() 
    print('\n ===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))+'\n') 

    f.close()

if __name__ == '__main__':
    main()
