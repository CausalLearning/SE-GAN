import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim
import os
import time
from utils.tools import *
from dataset.gta5_dataset import GTA5DataSet
from model.Networks import DeeplabMulti
import torch.backends.cudnn as cudnn

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():

    parser = argparse.ArgumentParser(description="SEGAN")
    
    #dataset
    parser.add_argument("--data_dir_source", type=str, default='/home/yonghao.xu/Data/SegmentationData/GTA5/',
                        help="source dataset path.")
    parser.add_argument("--data_list_source", type=str, default='./dataset/GTA5_imagelist_train.txt',
                        help="source dataset list file.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")
    parser.add_argument("--input_size_src", type=str, default='1280,720',
                        help="width and height of input src images.")                   
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes.")

    #network
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                        help="base learning rate.")
    parser.add_argument("--num_steps", type=int, default=250000,
                        help="Number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=150000,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--restore_from", type=str, default='/home/yonghao.xu/PreTrainedModel/DeepLab_resnet_pretrained_init-f81d91e8.pth',
                        help="pretrained ResNet model.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./PreTrain/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

def main():

    args = get_arguments()
    
    snapshot_dir = args.snapshot_dir+'GTA/ResNet_Pretrain_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    f = open(snapshot_dir+'GTASeg_log.txt', 'w')

    w, h = map(int, args.input_size_src.split(','))
    input_size_src = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Create network
    student_net = DeeplabMulti(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)

    new_params = student_net.state_dict().copy()
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if not args.num_classes == 19 or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            # print i_parts
    student_net.load_state_dict(new_params)

    student_net.train()
    student_net = student_net.cuda()
   
    src_loader = data.DataLoader(
                    GTA5DataSet(args.data_dir_source, args.data_list_source, max_iters=args.num_steps_stop*args.batch_size,
                    crop_size=input_size_src,
                    scale=False, mirror=False, mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.SGD(student_net.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    interp = nn.Upsample(size=(input_size_src[1], input_size_src[0]), mode='bilinear')
    loss_hist = np.zeros((args.num_steps_stop,2))

    for batch_index, src_data in enumerate(src_loader):

        tem_time = time.time()
        student_net.train()
        optimizer.zero_grad()
        
        adjust_learning_rate(optimizer,args.learning_rate,batch_index,args.num_steps)

        images, src_label, _, _ = src_data
        images = images.cuda()
        src_label = src_label.cuda()            
        _,src_output_ori = student_net(images)
        src_output = interp(src_output_ori)
        
        # Segmentation Loss
        cls_loss_value = loss_calc(src_output, src_label)
        _, predict_labels = torch.max(src_output, 1)
        lbl_pred = predict_labels.detach().cpu().numpy()
        lbl_true = src_label.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
            metrics_batch.append(mean_iu)                
        miou = np.mean(metrics_batch, axis=0)  
        
        total_loss = cls_loss_value
        
        loss_hist[batch_index,0] = total_loss.item()
        loss_hist[batch_index,1] = miou

        total_loss.backward()
        optimizer.step()
        
        batch_time = time.time()-tem_time

        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d time: %.2f miou = %.1f cls_loss = %.3f \n'%(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,1])*100,np.mean(loss_hist[batch_index-9:batch_index+1,0])))
            f.write('Iter %d/%d time: %.2f miou = %.1f cls_loss = %.3f \n'%(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,1])*100,np.mean(loss_hist[batch_index-9:batch_index+1,0])))
            f.flush() 
            
        if (batch_index+1) % 10000 == 0: 
            # Saving the models      
            f.write('Save Model\n') 
            print('Save Model\n')                     
            model_name = 'GTA_batch_'+repr(batch_index+1)+'_miou_'+repr(int(np.mean(loss_hist[batch_index-9:batch_index+1,1])*100))+'.pth'
            torch.save(student_net.state_dict(), os.path.join(
                snapshot_dir, model_name))   

    f.close()

if __name__ == '__main__':
    main()
