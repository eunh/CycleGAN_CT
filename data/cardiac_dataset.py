import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset_cardiac
from PIL import Image
import PIL
import random
import numpy as np
import scipy.io as sio
import torch

class Cardiac_LD(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.A_paths = make_dataset_cardiac(self.dir_AB, phase=[1])
        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)

        if self.opt.isTrain:
            self.B_paths = make_dataset_cardiac(self.dir_AB, phase=[8])
            self.B_paths = sorted(self.B_paths)
            self.B_size = len(self.B_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        A_img = sio.loadmat(A_path)['img']
        h, w = np.shape(A_img)
        A_img = np.reshape(A_img, (1,w,h)).astype(float)
        A_max = float(np.amax(A_img))
        A = (torch.from_numpy(A_img).float()/A_max-0.5)*2

        if self.opt.isTrain:
            B_path = self.B_paths[index_A]
            B_img = sio.loadmat(B_path)['img']
            B_img = np.reshape(B_img, (1,256,256)).astype(float)
            B = (torch.from_numpy(B_img).float()/A_max-0.5)*2

        if self.opt.isTrain:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            A = A[:,h_offset:h_offset + self.opt.fineSize,  w_offset:w_offset + self.opt.fineSize]
            B = B[:,h_offset:h_offset + self.opt.fineSize,  w_offset:w_offset + self.opt.fineSize]


        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if self.opt.isTrain:
            return {'A': A, 'B': B,
                    'A_paths': A_path, 'B_paths': B_path, 'A_max': A_max}
        else:
            return {'A': A, 'A_paths': A_path, 'A_max': A_max}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'UnalignedDataset'
