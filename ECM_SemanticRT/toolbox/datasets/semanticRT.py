import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class SemanticRT(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):

        assert mode in ['train', 'val', 'test', 'test_day', 'test_night', 'test_mc', 'test_mo', 'test_hard'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        # self.val_resize = Resize(crop_size)

        self.resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.681, 43.623, 41.695, 42.325, 38.371, 42.011, 6.873, 43.406, 40.634, 37.884, 37.325, 31.001, 30.114])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0098, 2.4369, 1.5490, 1.4971, 1.06759, 1.54816, 0.05627, 0.49118, 1.0, 0.7768, 1.0517, 0.6223, 0.56027])
        else:
            raise (f"{cfg['class_weight']} not support.")

        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()

        image = Image.open(os.path.join(self.root, 'rgb', image_path+'.jpg'))
        depth = Image.open(os.path.join(self.root, 'thermal', image_path+'.jpg'))
        depth = depth.convert('RGB')  #
        label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))
        label = label.convert('L')
        binary_label = Image.open(os.path.join(self.root, 'binary_labels', image_path + '.png'))
        binary_label = binary_label.convert('L')

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
            'binary_label': binary_label,
        }

        sample = self.resize(sample)  #

        if self.mode in ['train'] and self.do_aug:  
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['binary_label'] = torch.from_numpy(np.asarray(sample['binary_label'], dtype=np.int64) / 255.).long()

        sample['label_path'] = image_path.strip().split('/')[-1]  
        return sample

    @property
    def cmap(self):
        return [
    (0, 0, 0),          # 0: background (unlabeled)
    (72, 61, 39),       # 1: car_stop
    (0, 0, 255),        # 2: bike
    (148, 0, 211),      # 3: bicyclist
    (128, 128, 0),      # 4: motorcycle
    (64, 64, 128),      # 5: motorcyclist
    (0, 139, 139),      # 6: car
    (131, 139, 139),    # 7: tricycle
    (192, 64, 0),       # 8: traffic_light
    (126, 192, 238),    # 9: box
    (244, 164, 96),     # 10:pole
    (211, 211, 211),    # 11:curve
    (205, 155, 155),    # 12:person
        ]

if __name__ == '__main__':
    path = '/home/user/projects/SemanticRT/test/rgb'
    name = os.listdir(path)
    name.sort()
    save = '/home/user/projects/SemanticRT/test.txt'
  
