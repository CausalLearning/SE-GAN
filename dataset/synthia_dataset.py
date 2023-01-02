import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image

class synthiaDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None,crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, data_dir_tgstn='None'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(max_iters / len(self.img_ids))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        self.id_to_trainid = {1: 9, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8,
                              7: 5, 8: 12, 9: 7, 10: 10, 11: 15, 12: 14, 15: 6,
                              17: 11, 19: 13, 21: 3}

        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        if data_dir_tgstn != 'None':
            for name in self.img_ids:
                img_file = osp.join(data_dir_tgstn, "images/%s" % name)
                label_file = osp.join(self.root, "labels/%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1] 
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name

          
