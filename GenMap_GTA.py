import argparse
import numpy as np
import torch
from torch.utils import data
from model.Networks import DeeplabMulti
from dataset.cityscapes_dataset import cityscapesDataSet
import os
from utils.tools import *
import torch.nn as nn

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():

    parser = argparse.ArgumentParser(description="SEGAN")

    parser.add_argument("--data_dir", type=str, default='/home/yonghao.xu/Data/SegmentationData/Cityscapes/',
                        help="target dataset path.")
    parser.add_argument("--data_list", type=str, default='./dataset/cityscapes_labellist_val.txt',
                        help="target dataset list file.")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='/home/yonghao.xu/PreTrainedModel/GTA2Cityscapes_batch_40500_miou_485.pth',
                        help="restored model.")   
    parser.add_argument("--snapshot_dir", type=str, default='./Maps/GTA/',
                        help="Path to save result.")
    return parser.parse_args()

def main():
    
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    f = open(args.snapshot_dir+'Evaluation.txt', 'w')
    
    model = DeeplabMulti(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()
    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024,2048), mode='bilinear')

    for index, batch in enumerate(testloader):
        if (index+1) % 10 == 0:
            print('%d processd' % (index+1))
        image, _,_, name = batch
        _,output = model(image.cuda())
        output = interp(output).cpu().data[0].numpy()   
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        name = name[0].split('/')[-1]
        output_col.save('%s/%s_SEGAN.png' % (args.snapshot_dir, name.split('.')[0]))
    
    f.close()

if __name__ == '__main__':
    main()
