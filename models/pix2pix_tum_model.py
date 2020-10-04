import torch
from .base_model import BaseModel
from . import networks_tum as networks


class Pix2PixTUMModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_Dice', type=float, default=100.0, help='weight for Dice loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_Dice', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_S', 'fake_S']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'S']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, 
                                      opt.output_nc, 
                                      opt.ngf, 
                                      opt.netG, 
                                      opt.norm,
                                      not opt.no_dropout, 
                                      opt.init_type, 
                                      opt.init_gain, 
                                      self.gpu_ids)

        # define a discriminator 
        # conditional GANs need to take both input and output images 
        # Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:  
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, 
                                          opt.ndf, 
                                          opt.netD,
                                          opt.n_layers_D, 
                                          opt.norm, 
                                          opt.init_type, 
                                          opt.init_gain, 
                                          self.gpu_ids)

            # define the segmentation network
            self.netS = networks.define_S(opt.output_nc, 
                                          1, 
                                          opt.ngf, 
                                          opt.netG, 
                                          opt.norm,
                                          not opt.no_dropout, 
                                          opt.init_type, 
                                          opt.init_gain, 
                                          self.gpu_ids)

        if self.isTrain:

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionDice = networks.DiceLoss()

            # initialize optimizers 
            # schedulers will be automatically created by function <BaseModel.setup>
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_S)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_S = input['S'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_S = self.netS(self.fake_B)  # fake segmentation 

        # print('-------------------------------------------------------')
        # print('Real A', torch.min(self.real_A), torch.max(self.real_A))
        # print('Fake B', torch.min(self.fake_B), torch.max(self.fake_B))
        # print('Real B', torch.min(self.real_B), torch.max(self.real_B))
        # print('Fake S', torch.min(self.fake_S), torch.max(self.fake_S))
        # print('Real S', torch.min(self.real_S), torch.max(self.real_S))
        # print('-------------------------------------------------------')

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        # since we use conditional GANs, we need to feed both input and output to the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) 
        # stop backprop to the generator by detaching fake_B 
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # average the real and fake discriminator losses and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Third, S(G(A)) = true_S
        self.loss_G_Dice = self.criterionDice(self.fake_S, self.real_S) * self.opt.lambda_Dice

        # combine GAN loss, L1 loss and segmentation loss for the generator and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Dice
        self.loss_G.backward()

    def backward_S(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_S = self.netS(self.fake_B.detach())  # fake segmentation
        self.loss_S = self.criterionDice(fake_S, self.real_S)
        self.loss_S.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netS, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

        # update S 
        self.set_requires_grad(self.netS, True)  
        self.optimizer_S.zero_grad()        # set S's gradients to zero
        self.backward_S()                   # calculate graidents for S
        self.optimizer_S.step()             # udpate S's weights