import os
import SimpleITK as sitk
from data.base_dataset import BaseDataset, get_transform, get_params
from PIL import Image
import torch


class TUMDataset(BaseDataset):
    """A dataset class for MRI to US Dataset."""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dataset_dir = opt.dataroot
        self.transform = get_transform(opt)

        self.A_filenames = []
        self.B_filenames = []
        self.S_filenames = []
        
        for trial in os.listdir(self.dataset_dir):
            trail_dir = os.path.join(self.dataset_dir, trial)
            for patient in os.listdir(trail_dir):
                patient_dir = os.path.join(trail_dir, patient, 'preprocessed/rigid/cropped')
                A_path = os.path.join(patient_dir, 'mr.mhd')
                B_path = os.path.join(patient_dir, 'trus.mhd')
                S_path = os.path.join(patient_dir, 'mr_tree.mhd')

            if os.path.exists(A_path) and os.path.exists(B_path) and os.path.exists(S_path):
                self.A_filenames.append(os.path.join(patient_dir, 'mr.mhd'))
                self.B_filenames.append(os.path.join(patient_dir, 'trus.mhd'))
                self.S_filenames.append(os.path.join(patient_dir, 'mr_tree.mhd'))

        if not (len(self.A_filenames) == len(self.B_filenames) == len(self.S_filenames)):
            raise AssertionError('Dataset Length Mismatch')

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
                
    def __getitem__(self, index):
        
        A = Image.fromarray(sitk.GetArrayFromImage(sitk.ReadImage(self.A_filenames[index])))
        B = Image.fromarray(sitk.GetArrayFromImage(sitk.ReadImage(self.B_filenames[index])))
        S = Image.fromarray(sitk.GetArrayFromImage(sitk.ReadImage(self.S_filenames[index])))

        # apply the same transform to A, B and S
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), method=Image.NEAREST)
        S_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), method=Image.NEAREST, segmentation=True)

        A = A_transform(A)
        B = B_transform(B)
        S = S_transform(S)

        ones = torch.ones_like(S, dtype=int)
        minus_ones = torch.ones_like(S, dtype=int) * (-1)
        S = torch.where(S > 0, ones, minus_ones)

        return {'A': A, 'B': B, 'S':S}

    def __len__(self):
        return len(self.A_filenames)  
