import os
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
from util import task
#import random
#import numpy as np
import torch

class CreateDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_paths, self.img_size = make_dataset(opt.img_file, opt.img_dir)
        # provides random file for training and testing
        if opt.mask_dir != 'none':
            self.mask_paths, self.mask_size = make_dataset(os.path.join(opt.mask_dir, 'index.csv'), '')
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # load image
        img, img_path = self.load_img(index)
        # load mask
        mask, mask_param = self.load_mask(img, index)
        #print(mask_param)
        #return {'img': img, 'img_path': img_path, 'mask': mask}
        return {'img': img, 'img_path': img_path, 'mask': mask, 'mask_param': mask_param}

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.img_paths[index % self.img_size]
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path

    def load_mask(self, img, index):
        """Load different mask types for training and testing"""
        mask_type_index = torch.randint(0, len(self.opt.mask_type), (1,))
        mask_type = self.opt.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return task.center_mask(img)

        # random regular mask
        if mask_type == 1:
            return task.random_regular_mask(img)

        # random irregular mask
        if mask_type == 2:
            return task.random_irregular_mask(img)

        # external mask from "Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV18)"
        if mask_type == 3:
            if self.opt.isTrain:
                mask_index = torch.randint(0, self.mask_size, (1,))
                #mask_index = index
            else:
                #mask_index = random.randint(0, self.mask_size-1)
                mask_index = index % self.mask_size
            mask_pil = Image.open(self.mask_paths[mask_index][0]).convert('RGB')
            size = mask_pil.size[0]
            if size > mask_pil.size[1]:
                size = mask_pil.size[1]
            '''
            mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(10),
                                                 transforms.CenterCrop([size, size]),
                                                 transforms.Resize(self.opt.fineSize),
                                                 transforms.ToTensor()
                                                 ])
            '''
            mask_transform = transforms.Compose([transforms.Resize(self.opt.fineSize),
                                                 transforms.ToTensor()
                                                 ])
            mask = (mask_transform(mask_pil) == 0).float()
            mask_pil.close()

            #mask_param = np.array([self.mask_paths[mask_index][1], self.mask_paths[mask_index][2], self.mask_paths[mask_index][3]])
            mask_param = self.mask_paths[mask_index][1]

            return mask, mask_param


def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=not opt.no_shuffle, num_workers=int(opt.nThreads))

    return dataset


def get_transform(opt):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    if opt.isTrain:
        if opt.resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Resize(osize))
            transform_list.append(transforms.RandomCrop(fsize))
        elif opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(fsize))
        else:
            transform_list.append(transforms.Resize(fsize))
        if opt.use_augment:
            transform_list.append(transforms.ColorJitter(0.0, 0.0, 0.0, 0.0))
        if opt.use_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if opt.use_rotation:
            transform_list.append(transforms.RandomRotation(3))
        if not opt.no_cycle:
            transform_list.append(RandomeCycle())
    else:
        if opt.test_cycle:
            transform_list.append(RandomeCycle())
        transform_list.append(transforms.Resize(fsize))

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

class RandomeCycle(object):
    def __init__(self):
        pass

    def __call__(self, x):
        h = x.height
        w = x.width
        s = int(torch.randint(0, w, (1,))) # division point
        
        a = x.crop((0, 0, s, h))
        b = x.crop((s, 0, w, h))

        dst = Image.new('RGB', (w, h))
        dst.paste(b, (0, 0))
        dst.paste(a, (w - s, 0))

        return dst
