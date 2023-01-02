import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim
import os
import time
from utils.tools import *
from dataset.cityscapes16_dataset import cityscapes16DataSet
from model.Networks import DeeplabMulti
import torch.backends.cudnn as cudnn

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():

    parser = argparse.ArgumentParser(description="SEGAN")
    
    #dataset
    parser.add_argument("--data_dir_target", type=str, default='/home/yonghao.xu/Data/SegmentationData/Cityscapes/',
                        help="target dataset path.")
    parser.add_argument("--data_list_target", type=str, default='./dataset/cityscapes_labellist_train.txt',
                        help="target dataset list file.")
    parser.add_argument("--data_list_val", type=str, default='./dataset/cityscapes_labellist_val.txt',
                        help="val dataset list file.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")
    parser.add_argument("--input_size_tgt", type=str, default='1024,512',
                        help="width and height of input target images.")  
    parser.add_argument("--input_size_test", type=str, default='2048,1024',
                        help="width and height of test images.")                        
    parser.add_argument("--num_classes", type=int, default=16,
                        help="number of classes.")

    #network
    parser.add_argument("--batch_size", type=int, default=3,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5,
                        help="base learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--num_steps", type=int, default=20000,
                        help="Number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=20000,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--restore-from", type=str, default='/home/yonghao.xu/PreTrainedModel/Synthia2Cityscapes_batch_24500_miou_450.pth',
                        help="restored model.")   
    parser.add_argument("--weight_decay", type=float, default=5e-5,
                        help="regularisation parameter for L2-loss.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./Snap/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()


class seg_criterion(object):
    def __init__(self, num_classes,often_balance):
        self.num_classes = num_classes
        self.often_balance = often_balance
        self.class_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.often_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.max_value = 7

    def update_class_criterion(self,labels):
        weight = torch.FloatTensor(self.num_classes).zero_().cuda()
        weight += 1
        count = torch.FloatTensor(self.num_classes).zero_().cuda()
        often = torch.FloatTensor(self.num_classes).zero_().cuda()
        often += 1
        n, h, w = labels.shape
        for i in range(self.num_classes):
            count[i] = torch.sum(labels==i)
            if count[i] < 64*64*n: #small objective
                weight[i] = self.max_value
        if self.often_balance:
            often[count == 0] = self.max_value

        self.often_weight = 0.9 * self.often_weight + 0.1 * often 
        self.class_weight = weight * self.often_weight
        
        return nn.CrossEntropyLoss(weight = self.class_weight, ignore_index=255)

def main():

    args = get_arguments()

    snapshot_dir = args.snapshot_dir+'Synthia/SelfTrain_adv1e3'+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    f = open(snapshot_dir+'Synthia2Cityscapes_log.txt', 'w')

    w, h = map(int, args.input_size_tgt.split(','))
    input_size_tgt = (w, h)
    w, h = map(int, args.input_size_test.split(','))
    input_size_test = (w, h)

    # Create network
    selftrain_net = DeeplabMulti(num_classes=args.num_classes)
   
    saved_state_dict = torch.load(args.restore_from)
  
    selftrain_net.load_state_dict(saved_state_dict)
    
    selftrain_net = selftrain_net.cuda()
   
    tgt_loader = data.DataLoader(
                    cityscapes16DataSet(args.data_dir_target, args.data_list_target, max_iters=args.num_steps_stop*args.batch_size,           
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='selftrain',pseudo_path='adv1e3'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

    val_loader = data.DataLoader(
                    cityscapes16DataSet(args.data_dir_target, args.data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=1, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)

    optimizer = optim.SGD(selftrain_net.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp_tgt = nn.Upsample(size=(input_size_tgt[1], input_size_tgt[0]), mode='bilinear', align_corners=True)
 
    loss_hist = np.zeros((args.num_steps_stop,2))

    OA_hist = 0.2
    seg_crit = seg_criterion(args.num_classes,often_balance=True)
  
    cudnn.enabled = True
    cudnn.benchmark = True
    
    for batch_index, tgt_data in enumerate(tgt_loader):
        if batch_index>=args.num_steps_stop:
            break
        
        selftrain_net.train()
        tem_time = time.time()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,args.learning_rate,batch_index,args.num_steps)

        tgt_image, tgt_label, _, im_name = tgt_data
        tgt_image = tgt_image.cuda()
        tgt_label = tgt_label.long().cuda()            
        tgt_output_ori1,tgt_output_ori2 = selftrain_net(tgt_image)
        tgt_output1 = interp_tgt(tgt_output_ori1)
        tgt_output2 = interp_tgt(tgt_output_ori2)

        # L_seg
        seg_loss = seg_crit.update_class_criterion(tgt_label)
        cls_loss_value = seg_loss(tgt_output1, tgt_label)*0.1 + seg_loss(tgt_output2, tgt_label)
        _, predict_labels = torch.max(tgt_output2, 1)
        lbl_pred = predict_labels.detach().cpu().numpy()
        lbl_true = tgt_label.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
            metrics_batch.append(mean_iu)                
        miou = np.mean(metrics_batch, axis=0)  
        
        loss_hist[batch_index,0] = cls_loss_value.item()
        loss_hist[batch_index,1] = miou

        cls_loss_value.backward()

        optimizer.step()
       
        batch_time = time.time()-tem_time

        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d time: %.2f miou = %.1f cls_loss = %.3f \n'\
                %(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,1])*100,\
                    np.mean(loss_hist[batch_index-9:batch_index+1,0])))
            f.write('Iter %d/%d time: %.2f miou = %.1f cls_loss = %.3f \n'\
                %(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,1])*100,\
                    np.mean(loss_hist[batch_index-9:batch_index+1,0])))
            f.flush() 
            
        if (batch_index+1) % 500 == 0:                    
            OA_new = test_mIoU16(f,selftrain_net, val_loader, batch_index+1,input_size_test,print_per_batches=10)
            
            # Saving the models        
            if OA_new > OA_hist:    
                f.write('Save Model\n') 
                print('Save Model')                    
                
                model_name = 'Synthia2Cityscapes_batch_'+repr(batch_index+1)+'_miou_'+repr(int(OA_new*1000))+'.pth'
                torch.save(selftrain_net.state_dict(), os.path.join(
                    snapshot_dir, model_name))   
                OA_hist = OA_new

    f.close()   
    np.savez(snapshot_dir+'Synthia_loss.npz',loss_hist=loss_hist) 
    
if __name__ == '__main__':
    main()
