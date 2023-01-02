import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import time
from utils.tools import *
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from model.Networks import DeeplabMulti,Discriminator
import torch.backends.cudnn as cudnn

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():

    parser = argparse.ArgumentParser(description="SEGAN")
    
    #dataset
    parser.add_argument("--data_dir_source", type=str, default='/home/yonghao.xu/Data/SegmentationData/GTA5/',
                        help="source dataset path.")
    parser.add_argument("--data_dir_tgstn", type=str, default='./TGSTN/GTA/',
                        help="aug dataset path.")
    parser.add_argument("--data_list_source", type=str, default='./dataset/GTA5_imagelist_train.txt',
                        help="source dataset list file.")
    parser.add_argument("--data_dir_target", type=str, default='/home/yonghao.xu/Data/SegmentationData/Cityscapes/',
                        help="target dataset path.")
    parser.add_argument("--data_list_target", type=str, default='./dataset/cityscapes_labellist_train.txt',
                        help="target dataset list file.")
    parser.add_argument("--data_list_val", type=str, default='./dataset/cityscapes_labellist_val.txt',
                        help="val dataset list file.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")
    parser.add_argument("--input_size_src", type=str, default='1280,720',
                        help="width and height of input source images.")    
    parser.add_argument("--input_size_tgt", type=str, default='1024,512',
                        help="width and height of input target images.")  
    parser.add_argument("--input_size_test", type=str, default='2048,1024',
                        help="width and height of test images.")                        
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes.")

    #network
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5,
                        help="base learning rate.")
    parser.add_argument("--learning_rate_D", type=float, default=1e-5,
                        help="discriminator learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--num_steps", type=int, default=80000,
                        help="Number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=50000,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--restore_from", type=str, default='/home/yonghao.xu/PreTrainedModel/GTA_batch_150000_miou_55.pth',
                        help="pretrained VGG model.")
    parser.add_argument("--weight_decay", type=float, default=5e-5,
                        help="regularisation parameter for L2-loss.")

    #hyperparameters
    parser.add_argument("--teacher_alpha", type=float, default=0.999,
                        help="teacher alpha in EMA.")
    parser.add_argument("--st_weight", type=float, default=3,
                        help="self-ensembling weight.")
    parser.add_argument("--adv_weight", type=float, default=1e-3,
                        help="adversarial training weight.")

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
            if count[i] < 64*64*n:
                weight[i] = self.max_value
        if self.often_balance:
            often[count == 0] = self.max_value

        self.often_weight = 0.9 * self.often_weight + 0.1 * often 
        self.class_weight = weight * self.often_weight
        
        return nn.CrossEntropyLoss(weight = self.class_weight, ignore_index=255)

def main():

    args = get_arguments()
    
    snapshot_dir = args.snapshot_dir+'GTA/SEGAN_adv_weight_'+str(args.adv_weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    f = open(snapshot_dir+'GTA2Cityscapes_log.txt', 'w')

    w, h = map(int, args.input_size_src.split(','))
    input_size_src = (w, h)
    w, h = map(int, args.input_size_tgt.split(','))
    input_size_tgt = (w, h)
    w, h = map(int, args.input_size_test.split(','))
    input_size_test = (w, h)

    # Create network
    student_net = DeeplabMulti(num_classes=args.num_classes)
    teacher_net = DeeplabMulti(num_classes=args.num_classes)
    D_net1 = Discriminator(num_classes=args.num_classes)
    D_net2 = Discriminator(num_classes=args.num_classes)
    D_net1.train()
    D_net2.train()
        
    saved_state_dict = torch.load(args.restore_from)   
    
    student_net.load_state_dict(saved_state_dict)
    teacher_net.load_state_dict(saved_state_dict)    
    
    for name, param in teacher_net.named_parameters():
    
        param.requires_grad=False

    student_net = student_net.cuda()
    teacher_net = teacher_net.cuda()
    D_net1 = D_net1.cuda()
    D_net2 = D_net2.cuda()
   
    src_loader = data.DataLoader(
                    GTA5DataSet(args.data_dir_source, args.data_list_source, max_iters=int(np.ceil(args.num_steps_stop*args.batch_size/2.)),
                    crop_size=input_size_src,
                    scale=False, mirror=False, mean=IMG_MEAN,data_dir_tgstn=args.data_dir_tgstn),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    tgt_loader = data.DataLoader(
                    cityscapesDataSet(args.data_dir_target, args.data_list_target, max_iters=args.num_steps_stop*args.batch_size,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

    val_loader = data.DataLoader(
                    cityscapesDataSet(args.data_dir_target, args.data_list_val, max_iters=None,                  
                    crop_size=input_size_tgt,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='val'),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)
    
    optimizer = optim.SGD(student_net.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    optimizer_D1 = optim.Adam(D_net1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()
    optimizer_D2 = optim.Adam(D_net2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()
    
    student_params = list(student_net.parameters())
    teacher_params = list(teacher_net.parameters())

    teacher_optimizer = WeightEMA(
        teacher_params, 
        student_params,
        alpha=args.teacher_alpha,
    )

    interp_src = nn.Upsample(size=(input_size_src[1], input_size_src[0]),  mode='bilinear', align_corners=True)
    interp_tgt = nn.Upsample(size=(input_size_tgt[1], input_size_tgt[0]),  mode='bilinear', align_corners=True)
    n_class = args.num_classes
    loss_hist = np.zeros((args.num_steps_stop,7))

    OA_hist = 0.2
    seg_crit = seg_criterion(args.num_classes,often_balance=True)
    aug_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    cudnn.enabled = True
    cudnn.benchmark = True
    student_net.eval()
    teacher_net.eval()
    
    for batch_index, (src_data, tgt_data) in enumerate(zip(src_loader, tgt_loader)):
        if batch_index>=args.num_steps_stop:
            break
        
        decay_adv = (1 - batch_index/args.num_steps)
        tem_time = time.time()
        optimizer.zero_grad()
        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate(optimizer,args.learning_rate,batch_index,args.num_steps)
        adjust_learning_rate(optimizer_D1,args.learning_rate_D,batch_index,args.num_steps)
        adjust_learning_rate(optimizer_D2,args.learning_rate_D,batch_index,args.num_steps)

        # Train F_s
        for param in student_net.parameters():
            param.requires_grad = True
        for param in D_net1.parameters():
            param.requires_grad = False
        for param in D_net2.parameters():
            param.requires_grad = False
        
        src_s_input, src_label, _, im_name = src_data
        src_s_input = src_s_input.cuda()
        src_label = src_label.long().cuda()            
        src_output_ori1,src_output_ori2 = student_net(src_s_input)
        src_output1 = interp_src(src_output_ori1)
        src_output2 = interp_src(src_output_ori2)

        # L_seg
        seg_loss = seg_crit.update_class_criterion(src_label)
        cls_loss_value = seg_loss(src_output1, src_label)*0.1 + seg_loss(src_output2, src_label)
        _, predict_labels = torch.max(src_output2, 1)
        lbl_pred = predict_labels.detach().cpu().numpy()
        lbl_true = src_label.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
            metrics_batch.append(mean_iu)                
        miou = np.mean(metrics_batch, axis=0)  
        
        # L_con
        images, label_target,_, im_name = tgt_data
        images = images.cuda()
        label_target = label_target.cuda()

        tgt_s_output_ori1,tgt_s_output_ori2 = student_net(images)
        _,tgt_t_output_ori = teacher_net(images)
        
        tgt_t_output = interp_tgt(tgt_t_output_ori)
        tgt_s_output1 = interp_tgt(tgt_s_output_ori1)
        tgt_s_output2 = interp_tgt(tgt_s_output_ori2)
        
        tgt_t_predicts = F.softmax(tgt_t_output, dim=1).transpose(1, 2).transpose(2, 3)
        tgt_s_predicts = F.softmax(tgt_s_output2, dim=1).transpose(1, 2).transpose(2, 3)

        tgt_s_predicts = tgt_s_predicts.contiguous().view(-1,n_class)
        tgt_t_predicts = tgt_t_predicts.contiguous().view(-1,n_class)
        
        aug_loss_value = aug_loss(tgt_s_predicts, tgt_t_predicts)

        D_out1 = D_net1(F.softmax(tgt_s_output1, dim=1))
        D_out2 = D_net2(F.softmax(tgt_s_output2, dim=1))

        loss_adv_tgt1 = bce_loss(D_out1,
                                    Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

        loss_adv_tgt2 = bce_loss(D_out2,
                                    Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())

        loss_adv_G = (loss_adv_tgt1*0.1 + loss_adv_tgt2)
    
        # Weighted sum     
        total_loss = cls_loss_value + args.st_weight * aug_loss_value + args.adv_weight * loss_adv_G * decay_adv            
    
        loss_hist[batch_index,0] = total_loss.item()
        loss_hist[batch_index,1] = cls_loss_value.item()
        loss_hist[batch_index,2] = aug_loss_value.item()
        loss_hist[batch_index,3] = loss_adv_G.item()
        loss_hist[batch_index,6] = miou

        total_loss.backward()

        # train D_lab
        for param in D_net1.parameters():
            param.requires_grad = True
        for param in D_net2.parameters():
            param.requires_grad = True
        for param in student_net.parameters():
            param.requires_grad = False

        # L_adv
        # train with source
        pred1 = src_output1.detach()
        pred2 = src_output2.detach()
    
        D_out1 = D_net1(F.softmax(pred1, dim=1))
        D_out2 = D_net2(F.softmax(pred2, dim=1))

        loss_D1 = bce_loss(D_out1,
                            Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())*0.5

        loss_D2 = bce_loss(D_out2,
                            Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())*0.5

        loss_D1.backward()
        loss_D2.backward()

        loss_hist[batch_index,4] = loss_D1.data.cpu().numpy()
        loss_hist[batch_index,5] = loss_D2.data.cpu().numpy()

        # train with target
        pred_target1 = tgt_s_output1.detach()
        pred_target2 = tgt_s_output1.detach()

        D_out1 = D_net1(F.softmax(pred_target1, dim=1))
        D_out2 = D_net2(F.softmax(pred_target2, dim=1))

        loss_D1 = bce_loss(D_out1,
                            Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())*0.5

        loss_D2 = bce_loss(D_out2,
                            Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())*0.5

        loss_D1.backward()
        loss_D2.backward()

        loss_hist[batch_index,4] += loss_D1.data.cpu().numpy()
        loss_hist[batch_index,5] += loss_D2.data.cpu().numpy()

        optimizer.step()
        teacher_optimizer.step()   
        optimizer_D1.step() 
        optimizer_D2.step()

        batch_time = time.time()-tem_time

        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d time: %.2f miou = %.1f cls_loss = %.3f st_loss = %.3f g_loss = %.3f d1_loss = %.3f d2_loss = %.3f \n'\
                %(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,6])*100,\
                    np.mean(loss_hist[batch_index-9:batch_index+1,1]),np.mean(loss_hist[batch_index-9:batch_index+1,2]),\
                        np.mean(loss_hist[batch_index-9:batch_index+1,3]),np.mean(loss_hist[batch_index-9:batch_index+1,4]),np.mean(loss_hist[batch_index-9:batch_index+1,5])))
            f.write('Iter %d/%d time: %.2f miou = %.1f cls_loss = %.3f st_loss = %.3f g_loss = %.3f d1_loss = %.3f d2_loss = %.3f \n'\
                %(batch_index+1,args.num_steps,batch_time,np.mean(loss_hist[batch_index-9:batch_index+1,6])*100,\
                    np.mean(loss_hist[batch_index-9:batch_index+1,1]),np.mean(loss_hist[batch_index-9:batch_index+1,2]),\
                        np.mean(loss_hist[batch_index-9:batch_index+1,3]),np.mean(loss_hist[batch_index-9:batch_index+1,4]),np.mean(loss_hist[batch_index-9:batch_index+1,5])))
            f.flush() 
            
        if (batch_index+1) % 500 == 0:                    
            OA_new = test_mIoU(f,teacher_net, val_loader, batch_index+1,input_size_test,print_per_batches=10)
            
            # Saving the models        
            if OA_new > OA_hist:    
                f.write('Save Model\n') 
                print('Save Model')                     
                model_name = 'GTA2Cityscapes_batch_'+repr(batch_index+1)+'_miou_'+repr(int(OA_new*1000))+'.pth'
                torch.save(teacher_net.state_dict(), os.path.join(
                    snapshot_dir, model_name))      
                OA_hist = OA_new

    f.close()
    np.savez(snapshot_dir+'GTA_loss.npz',loss_hist=loss_hist) 
    
if __name__ == '__main__':
    main()
