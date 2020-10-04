import numpy as np
import os
import sys
import ntpath
import time
from . import util
from torch.utils.tensorboard import SummaryWriter
import polyaxon_helper


class Visualizer():

    def __init__(self, opt):
        self.opt = opt
        if opt.local == False:
            log_dir = polyaxon_helper.get_outputs_path()
        else:
            log_dir = r"C:/Users/arkha/Desktop/Study/Intern/DAAD_Project/runs"
        
        self.writer = SummaryWriter(log_dir)

    def log_images(self, visuals, epoch):
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            self.writer.add_image(label, image_numpy.transpose(2, 0, 1), epoch)

    def log_losses(self, epoch, losses):
        for key, value in losses.items():
            self.writer.add_scalar(key, value, epoch)
        