import torch
from .base_model import BaseModel
from . import network as network
from . import base_function as base_function
from . import external_function
from util import task
import itertools

import numpy as np

class sigss(BaseModel):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""
    def name(self):
        return "sigss"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--train_paths', type=str, default='two', help='training strategies with one path or two paths')
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=0.1, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')
            parser.add_argument('--lambda_f', type=float, default=100.0, help='weight for features loss')
        else:
            parser.add_argument('--use_z_mean', action='store_true', help='use z mean for test')
            parser.add_argument('-s', '--symmetry', type=int, default=[1, 1, 1, 1, 1], nargs='*', help='0: s~q(s|x), 1: s=E[q(s|x)], 2: s in [μ-3σ u+3σ]')
        parser.add_argument('--type_resblock', type=int, default=1, help='0: normal, 1: w/ circular padding, 2: w/ series-paralell dilated conv.')
        parser.add_argument('--no_use_one_D', action='store_true', help='D = D_rec')
        parser.add_argument('--no_global_D', action='store_true', help='use global discreminator')
        parser.add_argument('--use_agn', action='store_true', help='use additive Gaussian noise')
        parser.add_argument('--use_mean4rec', action='store_true', help='use mean of p(z|x) for rec. path, but sample from p(z|y)')
        parser.add_argument('--num_sym', type=int, default=5, help='number of symmetry types')
        parser.add_argument('--kappa', type=float, default=3.0, help='kappa for caluculating weight')
        parser.add_argument('--use_long_term_attn', action='store_true', help='use long tern attention')
        parser.add_argument('--rearrange_input', action='store_true', help='rearrange_input for 360IC')
        parser.add_argument('--use_spdc', action='store_true', help='use series-parallel dilated conv. for 360IC')
        parser.add_argument('--cut_pole', type=float, default=0.0, help='the angle [degree] not not use the region near the polse for L1 loss')
        parser.add_argument('--ps_sig', type=float, default=1.0, help='SD of p(s)')
        parser.add_argument('--fix_qs_sig', action='store_true', help='fix SD of q(s|x) to SD of p(s)')

        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)

        self.loss_names = ['kl_rec', 'kl_g', 'app_rec', 'app_g', 'ad_g', 'img_d', 'ad_rec', 'img_d_rec', 'feat']
        self.visual_names = ['img_m', 'img_c', 'img_truth', 'img_out', 'img_g', 'img_rec']
        self.value_names = ['mu_m', 'sigma_m', 'mu_post', 'sigma_post', 'mu_prior', 'sigma_prior']
        self.model_names = ['E', 'G', 'D', 'D_rec']
        self.distribution = []

        # define the inpainting model
        if opt.use_agn:
            if opt.num_sym != 0:
                print("cannot use AGN and symmetry control at the same time.")
                exit()
            L=6
        else:
            L=3
        self.net_E = network.define_e(ngf=32, z_nc=128, img_f=128, L=L, layers=5, norm='none', activation='LeakyReLU',
                                      init_type='orthogonal', type_resblock=opt.type_resblock, use_agn=opt.use_agn, num_sym=opt.num_sym, gpu_ids=opt.gpu_ids)
        self.net_G = network.define_g(ngf=32, z_nc=128, img_f=128, L=0, layers=5, output_scale=opt.output_scale,
                                      norm='instance', activation='LeakyReLU', use_attn=not opt.no_attn, use_long_term_attn=opt.use_long_term_attn, init_type='orthogonal', type_resblock=opt.type_resblock, use_agn=opt.use_agn, gpu_ids=opt.gpu_ids)
        # define the discriminator model
        self.net_D = network.define_d(ndf=32, img_f=128, layers=5, model_type='ResDis', init_type='orthogonal', type_resblock=opt.type_resblock, use_gd=not opt.no_global_D, gpu_ids=opt.gpu_ids)
        if opt.no_use_one_D:
            self.net_D_rec = network.define_d(ndf=32, img_f=128, layers=5, model_type='ResDis', init_type='orthogonal', type_resblock=opt.type_resblock, use_gd=not opt.no_global_D, gpu_ids=opt.gpu_ids)
        else:
            self.net_D_rec = self.net_D

        # define the symmetry control
        self.net_H = network.SymmetryControl()

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            self.floss = torch.nn.MSELoss()
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters()),
                        filter(lambda p: p.requires_grad, self.net_E.parameters())), lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                                filter(lambda p: p.requires_grad, self.net_D_rec.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulers
        self.setup(opt)

    def make_s2_weight(self, height, width, cut_pole=0.0):
        w_s2_h = np.zeros((height, 1)) # weight for vertical direction
        
        for i in range(height):
            th = (i + 1) / (height + 1) * np.pi
            if np.pi * cut_pole / 180 <= th <= np.pi * (180 - cut_pole) / 180:
                w_s2_h[i][0] = np.sin(th)
            else:
                w_s2_h[i][0] = 0.0
                
        w_s2_w = np.ones((1, width)) # extendig to horizontal direction
        w_s2 = torch.tensor(w_s2_h * w_s2_w, dtype=torch.float32).view(1, 1, height, width)

        return w_s2

    def getVMFweight(self, h, w, m_ph, m_th, k):

        ph = 2 * np.pi * np.arange(0, w) / w
        th = np.pi * np.arange(0, h) / (h-1)

        ph = ph.reshape(1, w) * np.ones((h, 1))
        th = th.reshape(h, 1) * np.ones((1, w))

        X = np.sin(th) * np.cos(ph)
        Y = np.sin(th) * np.sin(ph)
        Z = np.cos(th) 

        Xm = np.sin(m_th) * np.cos(m_ph)
        Ym = np.sin(m_th) * np.sin(m_ph)
        Zm = np.cos(m_th)

        C = np.stack([X, Y, Z], 0)
        M = np.stack([Xm, Ym, Zm], 0)

        # inner product between each position and center position
        C1 = C.reshape(3, w*h).T
        C2 = C1.dot(M)
        C3 = C2.reshape(h, w)

        val = np.exp(k*C3)

        return val

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']
        self.mask_param = input['mask_param']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])

        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img * 2 - 1
        self.img_m = self.mask * self.img_truth
        self.img_c = (1 - self.mask) * self.img_truth

        # rearrange input
        if self.opt.rearrange_input:
            if self.opt.num_sym != 0:
                print("cannot rearrange input for symmetry control.")
                exit()
            width = self.img_m.shape[3]
            self.center_mask = self.input['mask_param'][:,0] / 2 / np.pi * width #マスクの中心点
            for i in range(self.img_m.shape[0]):
                self.img_truth[i] = base_function.shift3d(self.img_truth[i], int(self.center_mask[i]))
                self.img_m[i] = base_function.shift3d(self.img_m[i], int(self.center_mask[i]))
                self.img_c[i] = base_function.shift3d(self.img_c[i], int(self.center_mask[i]))
                self.mask[i] = base_function.shift3d(self.mask[i], int(self.center_mask[i]))

        # define weight for symmetry control
        height = self.mask.shape[2]
        width = self.mask.shape[3]
        hight_m = height // 32
        width_m = width // 32
        hight_e = height // 8
        width_e = width // 8
        self.w_s2 = self.make_s2_weight(height, width, self.opt.cut_pole)

        w_m = []
        w_e = []
        for mp in self.mask_param:
            for j in range(len(mp)//3):
                if mp[3*j + 2] > 0.1:
                    w_m_0 = self.getVMFweight(hight_m, width_m, mp[3*j], mp[3*j + 1], self.opt.kappa).reshape(1,hight_m, width_m)
                    w_e_0 = self.getVMFweight(hight_e, width_e, mp[3*j], mp[3*j + 1], self.opt.kappa).reshape(1,hight_e, width_e)
                    if j== 0:
                        w_m.append(w_m_0)
                        w_e.append(w_e_0)
                    else:
                        w_m[-1] += w_m_0
                        w_e[-1] += w_e_0
        self.w_m =  torch.tensor(w_m, dtype=torch.float32)
        self.w_e =  torch.tensor(w_e, dtype=torch.float32)
        '''
        with open('0.csv','w') as f:
            for i in range(self.w_e.shape[2]):
                for j in range(self.w_e.shape[3]):
                    f.write("{},".format(self.w_e[3][0][i][j]))
                f.write('\n')
        exit()
        '''
        if len(self.gpu_ids) > 0:
            self.w_s2 = self.w_s2.cuda(self.gpu_ids[0])
            self.w_m = self.w_m.cuda(self.gpu_ids[0])
            self.w_e = self.w_e.cuda(self.gpu_ids[0])

        # get multiple scales image ground truth and mask for training
        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)
        self.scale_w_s2 = task.scale_pyramid(self.w_s2, self.opt.output_scale)
        self.scale_w_s2_2 = task.scale_pyramid(self.scale_w_s2[0], 3)# for feature loss

    def test(self):
        """Forward function used in test time"""
        
        # encoder process
        if self.opt.add_id:
            id = np.random.randint(100000000)
        else:
            id = None

        distribution, f = self.net_E(self.img_m)
        q_mean = distribution[-1][0]
        q_sigma = distribution[-1][1]
        if self.opt.fix_qs_sig:
            q_sigma = 0*q_sigma + self.opt.ps_sig
        q_distribution = torch.distributions.Normal(q_mean, q_sigma)
        scale_mask = task.scale_img(self.mask, size=[f[2].size(2), f[2].size(3)])

        # decoder process
        s_set = []
        for i in range(self.opt.nsampling):
            f_m=f[-1]
            f_e=f[2]
            z = 0

            # not use symmetry control
            if self.opt.num_sym == 0:
                if self.opt.use_z_mean:
                    z = q_distribution.mean
                else:
                    z = q_distribution.sample()
            # use symmetry control
            else:
                s = self.get_s_for_test(i, q_distribution)
                f_m = self.net_H(f_m, s, self.w_m)
                f_e = self.net_H(f_e, s, self.w_e)

                # resister s
                if i == 0:
                    ss = torch.cat([s, q_distribution.mean.detach(), q_distribution.stddev.detach()], dim=1)
                    s_set = ss
                else:
                    ss = torch.cat([s, q_distribution.mean.detach(), q_distribution.stddev.detach()], dim=1)
                    s_set = torch.cat([s_set, ss], dim=1)

            self.img_g, attn = self.net_G(z, f_m=f_m, f_e=f_e, mask=scale_mask.chunk(3, dim=1)[0])
            self.img_out = (1 - self.mask) * self.img_g[-1].detach() + self.mask * self.img_m
            #self.img_out = self.img_g[-1].detach()

            # rearrange input
            if self.opt.rearrange_input:
                for j in range(self.img_g[-1].shape[0]):
                    self.img_g[-1][j] = base_function.shift3d(self.img_g[-1][j], -int(self.center_mask[j]))
                    self.img_out[j] = base_function.shift3d(self.img_out[j], -int(self.center_mask[j]))

            self.score = self.net_D(self.img_out)
            self.save_results(self.img_out, i, data_name='out', id=id)
        
        if len(s_set) > 0:
            s_set = s_set.reshape(-1, s.shape[1]*3) #align by input images

        # rearrange input
        if self.opt.rearrange_input:
            for i in range(self.img_truth.shape[0]):
                self.img_truth[i] = base_function.shift3d(self.img_truth[i], -int(self.center_mask[i]))
                self.img_m[i] = base_function.shift3d(self.img_m[i], -int(self.center_mask[i]))
                
        # save the groundtruth and masked image
        self.save_results(self.img_truth, data_name='truth', id=id)
        self.save_results(self.img_m, data_name='mask', id=id)

        return s_set

    def get_s_for_test(self, i, q_distribution):
        # opt.symmetry = {0: s~q(s|x), 1: s=E[q(s|x)], 2: s in [μ-3σ u+3σ]}
        ch_s = len(self.opt.symmetry)
        s = q_distribution.sample()

        for k in range(ch_s):
            if self.opt.symmetry[k] == 1:
                s[:,k] = q_distribution.mean[:, k].detach()
            elif self.opt.symmetry[k] == 2:
                mid = (self.opt.nsampling - 1) / 2
                s[:, k] = q_distribution.mean[:, k].detach() + (i - mid) / mid * 3 * q_distribution.stddev[:, k].detach()
            elif self.opt.symmetry[k] != 0:
                print("--symmetry is invalid.")
                exit()
        
        return s

    def get_distribution(self, distributions):
        """Calculate encoder distribution for img_m, img_c"""
        # get distribution
        sum_valid = (torch.mean(self.mask.view(self.mask.size(0), -1), dim=1) - 1e-5).view(-1, 1, 1, 1)
        m_sigma = 1 / (1 + ((sum_valid - 0.8) * 8).exp_())
        p_distribution, q_distribution, kl_rec, kl_g = 0, 0, 0, 0

        self.distribution = []
        for distribution in distributions:
            p_mu, p_sigma, q_mu, q_sigma = distribution
            if self.opt.fix_qs_sig:
                q_sigma = 0*q_sigma + self.opt.ps_sig
            
            # the assumption distribution for different mask regions
            m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma))
            # m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_sigma))
            # the post distribution from mask regions
            p_distribution = torch.distributions.Normal(p_mu, p_sigma)
            p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_sigma.detach())
            # the prior distribution from valid region
            q_distribution = torch.distributions.Normal(q_mu, q_sigma)

            # kl divergence
            if self.opt.use_agn:
                kl_rec += torch.distributions.kl_divergence(m_distribution, p_distribution)
                if self.opt.train_paths == "one":
                    kl_g += torch.distributions.kl_divergence(m_distribution, q_distribution)
                elif self.opt.train_paths == "two":
                    kl_g += torch.distributions.kl_divergence(p_distribution_fix, q_distribution)
            else:
                p_distribution = torch.distributions.Normal(torch.zeros_like(q_mu), self.opt.ps_sig*torch.ones_like(q_sigma))
                #kl_rec = torch.exp(-0.5*(q_mu*q_mu).sum())/ np.sqrt((2*np.pi))**(q_mu.shape[0]*q_mu.shape[1])
                #kl_rec = ((q_mu-p_mu)*(q_mu-p_mu)/2/p_sigma/p_sigma).sum()
                kl_rec = ((q_mu-p_mu)*(q_mu-p_mu)/2/self.opt.ps_sig/self.opt.ps_sig).sum()
                kl_g = torch.distributions.kl_divergence(p_distribution, q_distribution)
                m_sigma = torch.ones_like(p_sigma)
            self.distribution.append([torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma), p_mu, p_sigma, q_mu, q_sigma])
            
        return p_distribution, q_distribution, kl_rec, kl_g

    def get_G_inputs(self, p_distribution, q_distribution, f):
        """Process the encoder feature and distributions for generation network"""
        f_m_rec = f[-1].chunk(2)[0]
        f_m_gen = f[-1].chunk(2)[0]
        f_e_rec = f[2].chunk(2)[0]
        f_e_gen = f[2].chunk(2)[0]
        scale_mask = task.scale_img(self.mask, size=[f_e_rec.size(2), f_e_rec.size(3)])
        mask = torch.cat([scale_mask.chunk(3, dim=1)[0], scale_mask.chunk(3, dim=1)[0]], dim=0)
        z = 0
        if self.opt.use_agn:
            if self.opt.use_mean4rec:
                z_p = q_distribution.mean
            else:
                z_p = p_distribution.rsample()
            z_q = q_distribution.rsample()
            z = torch.cat([z_p, z_q], dim=0)
            loss_feat = self.floss(f_m_rec, f_m_rec) # dummy
        elif self.opt.num_sym > 0:
            s_mean = q_distribution.mean
            s_rand = q_distribution.rsample()

            f_m_rec = self.net_H(f_m_rec, s_mean, self.w_m)
            f_m_gen = self.net_H(f_m_gen, s_rand, self.w_m)
            f_e_rec = self.net_H(f_e_rec, s_mean, self.w_e)
            f_e_gen = self.net_H(f_e_gen, s_rand, self.w_e)

            f_m_gt = f[-1].chunk(2)[1]
            f_e_gt = f[2].chunk(2)[1]
            loss_feat = self.floss(f_m_rec*self.scale_w_s2_2[0], f_m_gt*self.scale_w_s2_2[0])
            loss_feat += self.floss(f_e_rec*self.scale_w_s2_2[2], f_e_gt*self.scale_w_s2_2[2])
            loss_feat *= self.opt.lambda_f
        else:
            loss_feat = self.floss(f_m_rec, f_m_rec) # dummy

        f_m = torch.cat([f_m_rec, f_m_gen], dim=0)
        f_e = torch.cat([f_e_rec, f_e_gen], dim=0)

        return z, f_m, f_e, mask, loss_feat

    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        if self.opt.use_agn:
            distributions, f = self.net_E(self.img_m, self.img_c)
        else:
            distributions, f = self.net_E(self.img_m, self.img_truth)
        p_distribution, q_distribution, self.kl_rec, self.kl_g = self.get_distribution(distributions) #p(z|I_c), q(z|I_m)

        # decoder process
        z, f_m, f_e, mask, self.loss_feat = self.get_G_inputs(p_distribution, q_distribution, f)
        results, attn = self.net_G(z, f_m, f_e, mask)
        self.img_rec = []
        self.img_g = []
        for result in results:
            img_rec, img_g = result.chunk(2)
            self.img_rec.append(img_rec)
            self.img_g.append(img_g)
        self.img_out = (1-self.mask) * self.img_g[-1].detach() + self.mask * self.img_truth

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real_l, D_real_g = netD(real)
        D_real_loss = self.GANloss(D_real_l, True, True)
        if not self.opt.no_global_D:
            D_real_loss += self.GANloss(D_real_g, True, True)
        # fake
        D_fake_l, D_fake_g = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake_l, False, True)
        if not self.opt.no_global_D:
            D_fake_loss += self.GANloss(D_fake_g, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss +=gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D, self.net_D_rec)
        self.loss_img_d = self.backward_D_basic(self.net_D, self.img_truth, self.img_g[-1])
        self.loss_img_d_rec = self.backward_D_basic(self.net_D_rec, self.img_truth, self.img_rec[-1])

    def backward_G(self):
        """Calculate training loss for the generator"""

        # encoder kl loss
        self.loss_kl_rec = self.kl_rec.mean() * self.opt.lambda_kl * self.opt.output_scale
        self.loss_kl_g = self.kl_g.mean() * self.opt.lambda_kl * self.opt.output_scale

        # generator adversarial loss
        base_function._freeze(self.net_D, self.net_D_rec)
        # g loss fake
        D_fake_l, D_fake_g = self.net_D(self.img_g[-1])
        self.loss_ad_g = self.GANloss(D_fake_l, True, False) * self.opt.lambda_g
        if not self.opt.no_global_D:
            self.loss_ad_g += self.GANloss(D_fake_g, True, False) * self.opt.lambda_g

        # rec loss fake
        D_fake_l, D_fake_g = self.net_D_rec(self.img_rec[-1])
        D_real_l, D_real_g = self.net_D_rec(self.img_truth)
        self.loss_ad_rec = self.L2loss(D_fake_l, D_real_l) * self.opt.lambda_g
        if not self.opt.no_global_D:
            self.loss_ad_rec += self.L2loss(D_fake_g, D_real_g) * self.opt.lambda_g

        # calculate l1 loss ofr multi-scale outputs
        loss_app_rec, loss_app_g = 0, 0
        for i, (img_rec_i, img_fake_i, img_real_i, mask_i, w_s2_i) in enumerate(zip(self.img_rec, self.img_g, self.scale_img, self.scale_mask, self.scale_w_s2)):
            loss_app_rec += self.L1loss(img_rec_i*w_s2_i, img_real_i*w_s2_i)
            if self.opt.train_paths == "one":
                loss_app_g += self.L1loss(img_fake_i*w_s2_i, img_real_i*w_s2_i)
            elif self.opt.train_paths == "two":
                loss_app_g += self.L1loss(img_fake_i*mask_i*w_s2_i, img_real_i*mask_i*w_s2_i)
        self.loss_app_rec = loss_app_rec * self.opt.lambda_rec
        self.loss_app_g = loss_app_g * self.opt.lambda_rec

        # if one path during the training, just calculate the loss for generation path
        if self.opt.train_paths == "one":
            self.loss_app_rec = self.loss_app_rec * 0
            self.loss_ad_rec = self.loss_ad_rec * 0
            self.loss_kl_rec = self.loss_kl_rec * 0

        total_loss = 0

        for name in self.loss_names:
            if name != 'img_d' and name != 'img_d_rec':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        # compute the image completion results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
