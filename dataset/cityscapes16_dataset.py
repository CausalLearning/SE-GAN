import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image

class cityscapes16DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', pseudo_path=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 23: 9, 24: 10, 25: 11,
                              26: 12, 28: 13, 32: 14, 33: 15}
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            n_repeat = int(max_iters / len(self.img_ids))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
        self.files = []
        self.set = set
        if self.set=='selftrain':
            for name in self.img_ids:
                img_file = osp.join(self.root, "leftImg8bit/%s/%s" % ('train', name))              
                label_file = osp.join('./PseudoLabel/Synthia/'+pseudo_path, "%s" % (name.split('/')[-1].split('.')[0]+'_PseudoLabel.png'))
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name.split('/')[-1]
                })
        else:     
            for name in self.img_ids:
                img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
                label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name[:-15]+'gtFine_labelIds.png'))
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name.split('/')[-1]
                })
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        if self.set != 'val':
            label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        if self.set == 'train' or self.set == 'val':
            label_copy = 255 * np.ones(label.shape, dtype=np.float32)        
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            label_copy = label

        size = image.shape
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(),np.array(size), name
