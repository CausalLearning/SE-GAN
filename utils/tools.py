import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
import json

def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

class WeightEMA(object):
    def __init__(self, params, src_params, alpha):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)

def colorize_mask(mask):
    # mask: numpy array of the mask
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask    

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().cuda()
    criterion = CrossEntropy2d().cuda()
    return criterion(pred, label)

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class=19):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def test_mIoU(f,model, data_loader, epoch,input_size, print_per_batches=10):

    model.eval()
    num_classes = 19
    num_batches = len(data_loader)
    hist = np.zeros((num_classes, num_classes))    
    with open('./dataset/info.json','r') as fp:
      info = json.load(fp)
    name_classes = np.array(info['label'], dtype=np.str)
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    
    for ind, data in enumerate(data_loader):
        image, label = data[0].cuda(),data[1].squeeze()
            
        _,outputs = model(image)
        outputs = interp(outputs)
        _, predicted = torch.max(outputs, 1)

        pred = predicted.cpu().squeeze().numpy()
        label = label.numpy()

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:d}'.format(len(label.flatten()), len(pred.flatten()), ind))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % print_per_batches == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, num_batches, 100*np.mean(per_class_iu(hist))))
            f.write('{:d} / {:d}: {:0.2f}\n'.format(ind, num_batches, 100*np.mean(per_class_iu(hist)))) 
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        f.write('\n===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))) 
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    f.write('\n epoch %d ===> mIoU: ' %(epoch) + str(round(np.nanmean(mIoUs) * 100, 2))+'\n') 
    f.flush() 
    print('epoch %d ===> mIoU: ' %(epoch) + str(round(np.nanmean(mIoUs) * 100, 2)))
    return np.nanmean(mIoUs)

def test_mIoU16(f,model, data_loader, epoch,input_size, print_per_batches=10):

    model.eval()
    num_classes = 16
    num_batches = len(data_loader)
    hist = np.zeros((num_classes, num_classes))    
    with open('./dataset/info16.json','r') as fp:
      info = json.load(fp)
    name_classes = np.array(info['label'], dtype=np.str)
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    
    for ind, data in enumerate(data_loader):
        image, label = data[0].cuda(),data[1].squeeze()
            
        _,outputs = model(image)
        outputs = interp(outputs)
        _, predicted = torch.max(outputs, 1)

        pred = predicted.cpu().squeeze().numpy()
        label = label.numpy()

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:d}'.format(len(label.flatten()), len(pred.flatten()), ind))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % print_per_batches == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, num_batches, 100*np.mean(per_class_iu(hist))))
            f.write('{:d} / {:d}: {:0.2f}\n'.format(ind, num_batches, 100*np.mean(per_class_iu(hist)))) 
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        f.write('\n===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))) 
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    f.write('\n epoch %d ===> mIoU: ' %(epoch) + str(round(np.nanmean(mIoUs) * 100, 2))+'\n') 
    f.flush() 
    print('epoch %d ===> mIoU: ' %(epoch) + str(round(np.nanmean(mIoUs) * 100, 2)))
    return np.nanmean(mIoUs)

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()        
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss
