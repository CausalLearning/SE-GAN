import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import os
import time
from utils.tools import *
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from model.Networks import BaseNet,TransformerNet,Discriminator

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():
   
    parser = argparse.ArgumentParser(description="SEGAN")
    
    #dataset
    parser.add_argument("--data_dir_source", type=str, default='/home/yonghao.xu/Data/SegmentationData/GTA5/',
                        help="source dataset path.")
    parser.add_argument("--data_list_source", type=str, default='./dataset/GTA5_imagelist_train.txt',
                        help="source dataset list file.")
    parser.add_argument("--data_dir_target", type=str, default='/home/yonghao.xu/Data/SegmentationData/Cityscapes/',
                        help="target dataset path.")
    parser.add_argument("--data_list_target", type=str, default='./dataset/cityscapes_labellist_train.txt',
                        help="target dataset list file.")
    parser.add_argument("--input_size", type=str, default='1024,512',
                        help="width and height of input images.")    
    parser.add_argument("--input_size_target", type=str, default='2048,1024',
                        help="width and height of target images.")                        
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes.")

    #network
    parser.add_argument("--batch_size", type=int, default=2,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="base learning rate.")
    parser.add_argument("--learning_rate_D", type=float, default=5e-5,
                        help="discriminator learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--num_epoch", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--restore_from", type=str, default='/home/yonghao.xu/PreTrainedModel/VGG_pretrained_GTA5.pth',
                        help="pretrained Seg model.")
    parser.add_argument("--weight_decay", type=float, default=0.00005,
                        help="regularisation parameter for L2-loss.")

    #hyperparameters
    parser.add_argument("--semantic_weight", type=float, default=10,
                        help="semantic weight.")
    parser.add_argument("--perceptual_weight", type=float, default=1,
                        help="perceptual weight.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./TGSTN/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

def main():
    
    args = get_arguments()
    
    snapshot_dir = args.snapshot_dir+'GTA/semantic_weight_'+str(args.semantic_weight)+'_perceptual_weight_'+str(args.perceptual_weight)+'_lr'+str(args.learning_rate)+'_advlr'+str(args.learning_rate_D)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    f = open(snapshot_dir+'GTATrans_log.txt', 'w')

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    # Create network
    trans_net = TransformerNet()
    seg_net = BaseNet(num_classes=args.num_classes)    
    D_net = Discriminator(num_classes=3)

    seg_net.load_state_dict(torch.load(args.restore_from))  
    for name, param in seg_net.named_parameters():
        param.requires_grad=False

    trans_net = trans_net.cuda()
    seg_net = seg_net.cuda()
    D_net = D_net.cuda()

    src_loader = data.DataLoader(
                    GTA5DataSet(args.data_dir_source, args.data_list_source,
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    tgt_loader = data.DataLoader(
                    cityscapesDataSet(args.data_dir_target, args.data_list_target, max_iters=24964,                  
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN,
                    set='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

    num_batches = min(len(src_loader),len(tgt_loader))
    
    optimizer = optim.Adam(trans_net.parameters(),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()
      
    optimizer_D = optim.Adam(D_net.parameters(), lr=args.learning_rate_D, weight_decay=args.weight_decay)
    optimizer_D.zero_grad()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    num_steps = args.num_epoch*num_batches
    loss_hist = np.zeros((num_steps,6))
    index_i = -1

    aug_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    trans_net.train()
    D_net.train()
    seg_net.eval()
    for epoch in range(args.num_epoch):
        for batch_index, (src_data, tgt_data) in enumerate(zip(src_loader, tgt_loader)):
            index_i += 1

            tem_time = time.time()
            optimizer.zero_grad()
            adjust_learning_rate(optimizer,args.learning_rate,index_i,num_steps)
            adjust_learning_rate(optimizer_D,args.learning_rate_D,index_i,num_steps)
       
            # train with source
            for param in D_net.parameters():
                param.requires_grad = False
            for param in trans_net.parameters():
                param.requires_grad = True

            images, src_label, _, _ = src_data
            images = images.cuda()
            src_label = src_label.cuda()            
            trans_imgs = trans_net(images)
            src_feature,_ = seg_net(images)
            trans_feature,trans_output_ori = seg_net(trans_imgs)
            trans_output = interp(trans_output_ori)

            # Semantic Loss
            cls_loss_value = loss_calc(trans_output, src_label)
            _, predict_labels = torch.max(trans_output, 1)
            lbl_pred = predict_labels.detach().cpu().numpy()
            lbl_true = src_label.detach().cpu().numpy()
            metrics_batch = []
            for lt, lp in zip(lbl_true, lbl_pred):
                _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
                metrics_batch.append(mean_iu)                
            miou = np.mean(metrics_batch, axis=0)  

            # Perceptual Loss
            perceptual_loss_value = aug_loss(trans_feature,src_feature)
            
            D_trans = D_net(trans_imgs)
            L_trans = Variable(torch.FloatTensor(D_trans.data.size()).fill_(target_label)).cuda()

            loss_adv_G = bce_loss(D_trans,L_trans)

            # TOTAL LOSS       
            total_loss = args.semantic_weight * cls_loss_value + args.perceptual_weight * perceptual_loss_value + loss_adv_G
            
            loss_hist[index_i,0] = total_loss.item()
            loss_hist[index_i,1] = cls_loss_value.item()
            loss_hist[index_i,2] = perceptual_loss_value.item()
            loss_hist[index_i,3] = loss_adv_G.item()
            loss_hist[index_i,5] = miou

            total_loss.backward()
            optimizer.step()

            # train D
            # bring back requires_grad
            for param in D_net.parameters():
                param.requires_grad = True
            for param in trans_net.parameters():
                param.requires_grad = False
            optimizer_D.zero_grad()
                  
            D_source = D_net(images)
            L_source = Variable(torch.FloatTensor(D_source.data.size()).fill_(source_label)).cuda()

            trans_imgs = trans_net(images)
            D_trans = D_net(trans_imgs)     
            L_trans = Variable(torch.FloatTensor(D_trans.data.size()).fill_(source_label)).cuda()   

            images, _,_, _ = tgt_data
            images = images.cuda()            
            D_tgt = D_net(images)
            L_tgt = Variable(torch.FloatTensor(D_tgt.data.size()).fill_(target_label)).cuda()

            loss_adv_D = 0.5 * (bce_loss(D_trans,L_trans)\
                        +bce_loss(D_source,L_source))\
                        +bce_loss(D_tgt,L_tgt)

            loss_adv_D.backward()
            optimizer_D.step()
            loss_hist[index_i,4] = loss_adv_D.item()

            batch_time = time.time()-tem_time

            if (batch_index+1) % 10 == 0: 
                print('epoch %d/%d:  %d/%d time: %.2f miou = %.1f cls_loss = %.3f per_loss = %.3f g_loss = %.3f d_loss = %.3f \n'\
                    %(epoch+1, args.num_epoch,batch_index+1,num_batches,batch_time,np.mean(loss_hist[index_i-9:index_i+1,5])*100,\
                        np.mean(loss_hist[index_i-9:index_i+1,1]),np.mean(loss_hist[index_i-9:index_i+1,2]),\
                            np.mean(loss_hist[index_i-9:index_i+1,3]),np.mean(loss_hist[index_i-9:index_i+1,4])))
                f.write('epoch %d/%d:  %d/%d time: %.2f miou = %.1f cls_loss = %.3f per_loss = %.3f g_loss = %.3f d_loss = %.3f \n'\
                    %(epoch+1, args.num_epoch,batch_index+1,num_batches,batch_time,np.mean(loss_hist[index_i-9:index_i+1,5])*100,\
                        np.mean(loss_hist[index_i-9:index_i+1,1]),np.mean(loss_hist[index_i-9:index_i+1,2]),\
                            np.mean(loss_hist[index_i-9:index_i+1,3]),np.mean(loss_hist[index_i-9:index_i+1,4])))
                f.flush() 

        # Saving the models 
        f.write('Save Model\n') 
        print('Save Model')                     
        model_name = 'GTATrans_epoch_'+repr(epoch+1)+'_batch_'+repr(batch_index+1)+'_miou_'+repr(int(np.mean(loss_hist[index_i-9:index_i+1,5])*100))+'.pth'
        torch.save(trans_net.state_dict(), os.path.join(
            snapshot_dir, model_name))   
    f.close()
    
if __name__ == '__main__':
    main()
