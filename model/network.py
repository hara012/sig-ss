from .base_function import *
from .external_function import SpectralNorm
import torch.nn.functional as F


##############################################################################################################
# Network function
##############################################################################################################
def define_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', type_resblock=1, use_agn=False, num_sym=5, num_cov_sym=0, gpu_ids=[]):

    net = ResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord, type_resblock, use_agn, num_sym, num_cov_sym)

    return init_net(net, init_type, activation, gpu_ids)


def define_g(output_nc=3, ngf=64, z_nc=512, img_f=512, L=1, layers=5, norm='instance', activation='ReLU', output_scale=1,
             use_spect=True, use_coord=False, use_attn=True, use_long_term_attn=True, init_type='orthogonal', type_resblock=1, use_agn=True, gpu_ids=[]):

    net = ResGenerator(output_nc, ngf, z_nc, img_f, L, layers, norm, activation, output_scale, use_spect, use_coord, use_attn, use_long_term_attn, type_resblock, use_agn)

    return init_net(net, init_type, activation, gpu_ids)


def define_d(input_nc=3, ndf=64, img_f=512, layers=6, norm='none', activation='LeakyReLU', use_spect=True, use_coord=False,
             use_attn=True,  model_type='ResDis', init_type='orthogonal', type_resblock=1, use_gd=True, gpu_ids=[]):

    if model_type == 'ResDis':
        net = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn, type_resblock, use_gd)
    elif model_type == 'PatchDis':
        net = PatchDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn, type_resblock)

    return init_net(net, init_type, activation, gpu_ids)


#############################################################################################################
# Network structure
#############################################################################################################
class ResEncoder(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ngf=64, z_nc=128, img_f=1024, L=6, layers=6, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False, type_resblock=1, use_agn=False, num_sym=5, num_cov_sym=0):
        super(ResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L
        self.use_agn = use_agn
        self.num_sym = num_sym
        self.num_cov_sym = num_cov_sym

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        if type_resblock == 1:
            self.block0 = ResBlockEncoderOptimizedCP(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)
        else:
            self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            if type_resblock == 1:
                block = ResBlockCP(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            elif type_resblock == 2:
                block = ResBlockSPDC(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            else:
                block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # inference part
        if self.use_agn:
            for i in range(self.L):
                if type_resblock == 1:
                    block = ResBlockCP(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
                elif type_resblock == 2:
                    block = ResBlockSPDC(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
                else:
                    block = ResBlock(ngf * mult, ngf * mult, ngf *mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
                setattr(self, 'infer_prior' + str(i), block)

            if type_resblock == 1:
                self.posterior = ResBlockCP(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
                self.prior = ResBlockCP(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            elif type_resblock == 2:
                self.posterior = ResBlockSPDC(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
                self.prior = ResBlockSPDC(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            else:
                self.posterior = ResBlock(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
                self.prior = ResBlock(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)

        if self.num_sym > 0:
            for i in range(self.L):
                mult_prev = mult
                mult = mult * 2
                if type_resblock == 1:
                    block = ResBlockCP(ngf * mult_prev, ngf * mult, ngf *mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
                elif type_resblock == 2:
                    block = ResBlockSPDC(ngf * mult_prev, ngf * mult, ngf *mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
                else:
                    block = ResBlock(ngf * mult_prev, ngf * mult, ngf *mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
                setattr(self, 'infer_prior' + str(i), block)
            self.n_in_ps = ngf*mult*2
            self.postrior_s_mu = torch.nn.Linear(self.n_in_ps, self.num_sym)
            if self.num_cov_sym == 0:
                self.postrior_s_std = torch.nn.Linear(self.n_in_ps, self.num_sym)
            elif self.num_cov_sym > 0:
                self.postrior_s_std = torch.nn.Linear(self.n_in_ps, self.num_sym * num_cov_sym)
            else:
                self.postrior_s_std = torch.nn.Linear(self.n_in_ps, self.num_sym * (-num_cov_sym))

    def forward(self, img_m, img_c=None):
        """
        :param img_m: image with mask regions I_m
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """
        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        # during the training, we have two paths, during the testing, we only have one paths
        
        if type(img_c) != type(None):
            distribution = self.two_paths(out)
            return distribution, feature
        else:
            distribution = self.one_path(out)
            return distribution, feature

    def one_path(self, f_in):
        """one path for baseline training or testing"""
        f_m = f_in
        distribution = []

        if self.use_agn:
            # infer state
            for i in range(self.L):
                infer_prior = getattr(self, 'infer_prior' + str(i))
                f_m = infer_prior(f_m)

            # get distribution
            o = self.prior(f_m)
            q_mu, q_std = torch.split(o, self.z_nc, dim=1)
            distribution.append([q_mu, F.softplus(q_std)])
        elif self.num_sym > 0:
            # infer state
            for i in range(self.L):
                infer_prior = getattr(self, 'infer_prior' + str(i))
                f_m = infer_prior(f_m)

            # get distribution
            s_mu = self.postrior_s_mu(f_m.view(-1,self.n_in_ps))
            
            if self.num_cov_sym == 0:
                s_std = F.softplus(self.postrior_s_std(f_m.view(-1,self.n_in_ps)))
                s_std = F.softplus(s_std)
            elif self.num_cov_sym > 0:
                basis_cov = self.postrior_s_std(f_m.view(-1,self.n_in_ps))
                basis_cov = basis_cov.reshape(-1, self.num_sym, self.num_cov_sym)
                s_std = torch.bmm(basis_cov, basis_cov.transpose(1, 2))
            else:
                basis_cov = self.postrior_s_std(f_m.view(-1,self.n_in_ps))
                basis_cov = basis_cov.reshape(-1, self.num_sym, -self.num_cov_sym)
                for i in range(-self.num_cov_sym):
                    elem = torch.diag_embed(basis_cov[:,:,i], offset=0, dim1=-2, dim2=-1)
                    if i==0:
                        s_std_half = elem
                    else:
                        s_std_half += torch.roll(elem, i, dims=2) + torch.roll(elem, i, dims=1)
                s_std = torch.bmm(s_std_half, s_std_half)

            distribution.append([s_mu, s_std])
        else:
            distribution.append([torch.zeros_like(f_m), torch.ones_like(f_m)])

        return distribution

    def two_paths(self, f_in):
        """two paths for the training"""
        f_m, f_c = f_in.chunk(2)
        distributions = []

        if self.use_agn:
            # get distribution
            o = self.posterior(f_c)
            p_mu, p_std = torch.split(o, self.z_nc, dim=1)
            distribution = self.one_path(f_m)
            distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])
        elif self.num_sym > 0:
            distribution = self.one_path(f_m)
            if self.num_cov_sym == 0:
                distributions.append([torch.zeros_like(distribution[0][0]), torch.ones_like(distribution[0][1]), distribution[0][0], distribution[0][1]]) # 1st and 2nd temrs are dummy.
            else:
                cov = torch.empty_like(distribution[0][1])
                cov[:] = torch.eye(cov.shape[1])
                distributions.append([torch.zeros_like(distribution[0][0]), cov, distribution[0][0], distribution[0][1]]) # 1st and 2nd temrs are dummy.
        else:
            distributions.append([torch.zeros_like(f_m), torch.ones_like(f_m), torch.zeros_like(f_m), torch.ones_like(f_m)]) #dummy

        return distributions

class ResGenerator(nn.Module):
    """
    ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """
    def __init__(self, output_nc=3, ngf=64, z_nc=128, img_f=1024, L=1, layers=6, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_coord=False, use_attn=True, use_long_term_attn=True, type_resblock=1, use_agn=True):
        super(ResGenerator, self).__init__()

        self.layers = layers
        self.L = L
        self.output_scale = output_scale
        self.use_attn = use_attn
        self.use_long_term_attn = use_long_term_attn
        self.use_agn = use_agn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # latent z to feature
        mult = min(2 ** (layers-1), img_f // ngf)
        if type_resblock == 1:
            self.generator = ResBlockCP(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
        elif type_resblock == 2:
            self.generator = ResBlockSPDC(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
        else:
            self.generator = ResBlock(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)

        # transform
        if use_agn:
            for i in range(self.L):
                if type_resblock == 1:
                    block = ResBlockCP(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
                elif type_resblock == 2:
                    block = ResBlockSPDC(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
                else:
                    block = ResBlock(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
                setattr(self, 'generator' + str(i), block)

        # decoder part
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 1), img_f // ngf)
            if i > layers - output_scale:
                # upconv = ResBlock(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                if type_resblock == 1:
                    upconv = ResBlockDecoderCP(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
                else:
                    upconv = ResBlockDecoder(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            else:
                # upconv = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                if type_resblock == 1:
                    upconv = ResBlockDecoderCP(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
                else:
                    upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            if i > layers - output_scale - 1:
                outconv = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect, use_coord)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i == 1 and use_attn:
                attn = Auto_Attn(ngf*mult, None)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, z, f_m=None, f_e=None, mask=None):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """

        if self.use_agn:
            f = self.generator(z)
            for i in range(self.L):
                generator = getattr(self, 'generator' + str(i))
                f = generator(f)
            out = f_m + f
        else:
            out = f_m

        # the features come from mask regions and valid regions, we directly add them together
        
        results= []
        attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if i == 1 and self.use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out, f_e, mask, self.use_long_term_attn)
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                results.append(output)
                out = torch.cat([out, output], dim=1)

        return results, attn


class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=True, type_resblock=1, use_gd=True, layers_gd=4):
        super(ResDiscriminator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn
        self.use_gd = use_gd
        self.layers_gd = layers_gd

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        if type_resblock == 1:
            self.block0 = ResBlockEncoderOptimizedCP(input_nc, ndf,norm_layer, nonlinearity, use_spect, use_coord)
        else:
            self.block0 = ResBlockEncoderOptimized(input_nc, ndf,norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm_layer)
                setattr(self, 'attn' + str(i), attn)
            if type_resblock == 1:
                block = ResBlockCP(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            else:
                block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        if type_resblock == 1:
            self.block1 = ResBlockCP(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        else:
            self.block1 = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 3))
       
        #global disc
        if use_gd:
            for i in range(layers_gd-1):
                if type_resblock == 1:
                    block = ResBlockCP(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'down', use_spect, use_coord)
                    self.global4 = ResBlockCP(ndf * mult, 1, ndf * mult, norm_layer, nonlinearity, 'down', use_spect, use_coord)
                else:
                    block = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'down', use_spect, use_coord)
                setattr(self, 'gd' + str(i), block)

            if type_resblock == 1:
                block = ResBlockCP(ndf * mult, 1, ndf * mult, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            else:
                block = ResBlock(ndf * mult, 1, ndf * mult, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'gd' + str(layers_gd-1), block)

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out_l = self.block1(out)
        out_l = self.conv(self.nonlinearity(out_l))

        if self.use_gd:
            out_g = out
            for i in range(self.layers_gd):
                gd = getattr(self, 'gd' + str(i))
                out_g = gd(out_g)
            return out_l, out_g
        else:
            return out_l, 0


class PatchDiscriminator(nn.Module):
    """
    Patch Discriminator Network for Local 70*70 fake/real
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest channel for the model
    :param layers: down sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectral normalization or not
    :param use_coord: use CoordConv or nor
    :param use_attn: use short+long attention or not
    """
    def __init__(self, input_nc=3, ndf=64, img_f=512, layers=3, norm='batch', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=False):
        super(PatchDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}
        sequence = [
            coord_conv(input_nc, ndf, use_spect, use_coord, **kwargs),
            nonlinearity,
        ]

        mult = 1
        for i in range(1, layers):
            mult_prev = mult
            mult = min(2 ** i, img_f // ndf)
            sequence +=[
                coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
                nonlinearity,
            ]

        mult_prev = mult
        mult = min(2 ** i, img_f // ndf)
        kwargs = {'kernel_size': 4, 'stride': 1, 'padding': 1, 'bias': False}
        sequence += [
            coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
            nonlinearity,
            coord_conv(ndf * mult, 1, use_spect, use_coord, **kwargs),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out

class SymmetryControl(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm = nn.Sigmoid()

    def forward(self, x, s, w):
        
        s = self.norm(s)   
        s = s.view(s.shape[0], -1, 1, 1)
        
        if s.shape[1] == 2:
            step = x.shape[3]//8
            x180 = shift(x, step*4)
            x180f = flip(x, step*4-1)
            w180 = s[:,0:1,:,:] * shift(w, step*4)
            w180f = s[:,1:2,:,:] * flip(w, step*4-1)
            a = w * x + w180 * x180 + w180f * x180f
            b = w + w180 + w180f
        elif s.shape[1] == 3:
            step = x.shape[3]//8
            x90 = shift(x, step*2)
            x180 = shift(x, step*4)
            x270 = shift(x, step*6)
            w90 = s[:,0:1,:,:] * shift(w, step*2)
            w180 = s[:,1:2,:,:] * shift(w, step*4)
            w270 = s[:,2:3,:,:] * shift(w, step*6)
            a = w * x + w90 * x90 + w180 * x180 + w270 * x270
            b = w + w90 + w180 + w270
        elif s.shape[1] == 4:
            step = x.shape[3]//8
            x90 = shift(x, step*2)
            x180 = shift(x, step*4)
            x270 = shift(x, step*6)
            x90f = flip(x, step*2-1)
            x180f = flip(x, step*4-1)
            w90 = s[:,0:1,:,:] * shift(w, step*2)
            w180 = s[:,1:2,:,:] * shift(w, step*4)
            w270 = s[:,0:1,:,:] * shift(w, step*6)
            w90f = s[:,2:3,:,:] * flip(w, step*2-1)
            w180f = s[:,3:4,:,:] * flip(w, step*4-1)
            a = w * x + w90 * x90 + w180 * x180 + w270 * x270 + w90f * x90f + w180f * x180f
            b = w + w90 + w180 + w270 + w90f + w180f
        elif s.shape[1] == 5:
            step = x.shape[3]//8
            x90 = shift(x, step*2)
            x180 = shift(x, step*4)
            x270 = shift(x, step*6)
            x90f = flip(x, step*2-1)
            x180f = flip(x, step*4-1)
            w90 = s[:,0:1,:,:] * shift(w, step*2)
            w180 = s[:,1:2,:,:] * shift(w, step*4)
            w270 = s[:,2:3,:,:] * shift(w, step*6)
            w90f = s[:,3:4,:,:] * flip(w, step*2-1)
            w180f = s[:,4:5,:,:] * flip(w, step*4-1)
            a = w * x + w90 * x90 + w180 * x180 + w270 * x270 + w90f * x90f + w180f * x180f
            b = w + w90 + w180 + w270 + w90f + w180f
        elif s.shape[1] == 11:
            step = x.shape[3]//8
            x45 = shift(x, step)
            x90 = shift(x, step*2)
            x135 = shift(x, step*3)
            x180 = shift(x, step*4)
            x225 = shift(x, step*5)
            x270 = shift(x, step*6)
            x315 = shift(x, step*7)
            x45f = flip(x, step-1)
            x90f = flip(x, step*2-1)
            x135f = flip(x, step*3-1)
            x180f = flip(x, step*4-1)
            w45 = s[:,0:1,:,:] * shift(w, step)
            w90 = s[:,1:2,:,:] * shift(w, step*2)
            w135 = s[:,2:3,:,:] * shift(w, step*3)
            w180 = s[:,3:4,:,:] * shift(w, step*4)
            w225 = s[:,4:5,:,:] * shift(w, step*5)
            w270 = s[:,5:6,:,:] * shift(w, step*6)
            w315 = s[:,6:7,:,:] * shift(w, step*7)
            w45f = s[:,7:8,:,:] * flip(w, step-1)
            w90f = s[:,8:9,:,:] * flip(w, step*2-1)
            w135f = s[:,9:10,:,:] * flip(w, step*3-1)
            w180f = s[:,10:11,:,:] * flip(w, step*4-1)
            a = w * x + w45 * x45 + w90 * x90 + w135 * x135 + w180 * x180 + w225 * x225 + w270 * x270 + w315 * x315 + w45f * x45f + w90f * x90f + w135f * x135f + w180f * x180f
            b = w + w45 + w90 + w135 + w180 + w225 + w270 + w315 + w45f + w90f + w135f + w180f
        elif s.shape[1] == 23:
            step = x.shape[3]//16
            x1 = shift(x, step)
            x2 = shift(x, step*2)
            x3 = shift(x, step*3)
            x4 = shift(x, step*4)
            x5 = shift(x, step*5)
            x6 = shift(x, step*6)
            x7 = shift(x, step*7)
            x8 = shift(x, step*8)
            x9 = shift(x, step*9)
            x10 = shift(x, step*10)
            x11 = shift(x, step*11)
            x12 = shift(x, step*12)
            x13 = shift(x, step*13)
            x14 = shift(x, step*14)
            x15 = shift(x, step*15)
            x1f = flip(x, step-1)
            x2f = flip(x, step*2-1)
            x3f = flip(x, step*3-1)
            x4f = flip(x, step*4-1)
            x5f = flip(x, step*5-1)
            x6f = flip(x, step*6-1)
            x7f = flip(x, step*7-1)
            x8f = flip(x, step*8-1)
            w1 = s[:,0:1,:,:] * shift(w, step)
            w2 = s[:,1:2,:,:] * shift(w, step*2)
            w3 = s[:,2:3,:,:] * shift(w, step*3)
            w4 = s[:,3:4,:,:] * shift(w, step*4)
            w5 = s[:,4:5,:,:] * shift(w, step*5)
            w6 = s[:,5:6,:,:] * shift(w, step*6)
            w7 = s[:,6:7,:,:] * shift(w, step*7)
            w8 = s[:,7:8,:,:] * shift(w, step*8)
            w9 = s[:,8:9,:,:] * shift(w, step*9)
            w10 = s[:,9:10,:,:] * shift(w, step*10)
            w11 = s[:,10:11,:,:] * shift(w, step*11)
            w12 = s[:,11:12,:,:] * shift(w, step*12)
            w13 = s[:,12:13,:,:] * shift(w, step*13)
            w14 = s[:,13:14,:,:] * shift(w, step*14)
            w15 = s[:,14:15,:,:] * shift(w, step*15)
            w1f = s[:,15:16,:,:] * flip(w, step-1)
            w2f = s[:,16:17,:,:] * flip(w, step*2-1)
            w3f = s[:,17:18,:,:] * flip(w, step*3-1)
            w4f = s[:,18:19,:,:] * flip(w, step*4-1)
            w5f = s[:,19:20,:,:] * flip(w, step*5-1)
            w6f = s[:,20:21,:,:] * flip(w, step*6-1)
            w7f = s[:,21:22,:,:] * flip(w, step*7-1)
            w8f = s[:,22:23,:,:] * flip(w, step*8-1)
            a = w * x + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7 + w8 * x8 + w9 * x9 + w10 * x10 + w11 * x11 + w12 * x12 + w13 * x13 + w14 * x14 + w15 * x15 + w1f * x1f + w2f * x2f  + w3f * x3f + w4f * x4f + w5f * x5f + w6f * x6f + w7f * x7f + w8f * x8f
            b = w + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10 + w11 + w12 + w13+ w14 + w15 + w1f + w2f + w3f + w4f + w5f + w6f + w7f + w8f
        else:
            print("num of symmetry is invalid.")
            exit()

        return a / b


class SymmetryControl2(nn.Module):
    def __init__(self):
        super().__init__()

        a_s = torch.ones(1)
        b_s = torch.zeros(1)
        self.w = torch.nn.Parameter(a_s)
        self.b = torch.nn.Parameter(b_s)

        self.norm = nn.Sigmoid()

    def forward(self, x, s, w):
        
        print(self.w, self.b)
        print('---')
        print(s)
        s = self.norm(self.w*s + self.b) 
        print(self.w)
        print(self.b)
        print(s)
        s = s.view(s.shape[0], -1, 1, 1)
        
        if s.shape[1] == 4:
            step = x.shape[3]//8
            x90 = shift(x, step*2)
            x180 = shift(x, step*4)
            x270 = shift(x, step*6)
            x90f = flip(x, step*2-1)
            x180f = flip(x, step*4-1)
            w90 = s[:,0:1,:,:] * shift(w, step*2)
            w180 = s[:,1:2,:,:] * shift(w, step*4)
            w270 = s[:,0:1,:,:] * shift(w, step*6)
            w90f = s[:,2:3,:,:] * flip(w, step*2-1)
            w180f = s[:,3:4,:,:] * flip(w, step*4-1)
            a = w * x + w90 * x90 + w180 * x180 + w270 * x270 + w90f * x90f + w180f * x180f
            b = w + w90 + w180 + w270 + w90f + w180f
        elif s.shape[1] == 5:
            step = x.shape[3]//8
            x90 = shift(x, step*2)
            x180 = shift(x, step*4)
            x270 = shift(x, step*6)
            x90f = flip(x, step*2-1)
            x180f = flip(x, step*4-1)
            w90 = s[:,0:1,:,:] * shift(w, step*2)
            w180 = s[:,1:2,:,:] * shift(w, step*4)
            w270 = s[:,2:3,:,:] * shift(w, step*6)
            w90f = s[:,3:4,:,:] * flip(w, step*2-1)
            w180f = s[:,4:5,:,:] * flip(w, step*4-1)
            a = w * x + w90 * x90 + w180 * x180 + w270 * x270 + w90f * x90f + w180f * x180f
            b = w + w90 + w180 + w270 + w90f + w180f
        elif s.shape[1] == 11:
            step = x.shape[3]//8
            x45 = shift(x, step)
            x90 = shift(x, step*2)
            x135 = shift(x, step*3)
            x180 = shift(x, step*4)
            x225 = shift(x, step*5)
            x270 = shift(x, step*6)
            x315 = shift(x, step*7)
            x45f = flip(x, step-1)
            x90f = flip(x, step*2-1)
            x135f = flip(x, step*3-1)
            x180f = flip(x, step*4-1)
            w45 = s[:,0:1,:,:] * shift(w, step)
            w90 = s[:,1:2,:,:] * shift(w, step*2)
            w135 = s[:,2:3,:,:] * shift(w, step*3)
            w180 = s[:,3:4,:,:] * shift(w, step*4)
            w225 = s[:,4:5,:,:] * shift(w, step*5)
            w270 = s[:,5:6,:,:] * shift(w, step*6)
            w315 = s[:,6:7,:,:] * shift(w, step*7)
            w45f = s[:,7:8,:,:] * flip(w, step-1)
            w90f = s[:,8:9,:,:] * flip(w, step*2-1)
            w135f = s[:,9:10,:,:] * flip(w, step*3-1)
            w180f = s[:,10:11,:,:] * flip(w, step*4-1)
            a = w * x + w45 * x45 + w90 * x90 + w135 * x135 + w180 * x180 + w225 * x225 + w270 * x270 + w315 * x315 + w45f * x45f + w90f * x90f + w135f * x135f + w180f * x180f
            b = w + w45 + w90 + w135 + w180 + w225 + w270 + w315 + w45f + w90f + w135f + w180f
        elif s.shape[1] == 23:
            step = x.shape[3]//16
            x1 = shift(x, step)
            x2 = shift(x, step*2)
            x3 = shift(x, step*3)
            x4 = shift(x, step*4)
            x5 = shift(x, step*5)
            x6 = shift(x, step*6)
            x7 = shift(x, step*7)
            x8 = shift(x, step*8)
            x9 = shift(x, step*9)
            x10 = shift(x, step*10)
            x11 = shift(x, step*11)
            x12 = shift(x, step*12)
            x13 = shift(x, step*13)
            x14 = shift(x, step*14)
            x15 = shift(x, step*15)
            x1f = flip(x, step-1)
            x2f = flip(x, step*2-1)
            x3f = flip(x, step*3-1)
            x4f = flip(x, step*4-1)
            x5f = flip(x, step*5-1)
            x6f = flip(x, step*6-1)
            x7f = flip(x, step*7-1)
            x8f = flip(x, step*8-1)
            w1 = s[:,0:1,:,:] * shift(w, step)
            w2 = s[:,1:2,:,:] * shift(w, step*2)
            w3 = s[:,2:3,:,:] * shift(w, step*3)
            w4 = s[:,3:4,:,:] * shift(w, step*4)
            w5 = s[:,4:5,:,:] * shift(w, step*5)
            w6 = s[:,5:6,:,:] * shift(w, step*6)
            w7 = s[:,6:7,:,:] * shift(w, step*7)
            w8 = s[:,7:8,:,:] * shift(w, step*8)
            w9 = s[:,8:9,:,:] * shift(w, step*9)
            w10 = s[:,9:10,:,:] * shift(w, step*10)
            w11 = s[:,10:11,:,:] * shift(w, step*11)
            w12 = s[:,11:12,:,:] * shift(w, step*12)
            w13 = s[:,12:13,:,:] * shift(w, step*13)
            w14 = s[:,13:14,:,:] * shift(w, step*14)
            w15 = s[:,14:15,:,:] * shift(w, step*15)
            w1f = s[:,15:16,:,:] * flip(w, step-1)
            w2f = s[:,16:17,:,:] * flip(w, step*2-1)
            w3f = s[:,17:18,:,:] * flip(w, step*3-1)
            w4f = s[:,18:19,:,:] * flip(w, step*4-1)
            w5f = s[:,19:20,:,:] * flip(w, step*5-1)
            w6f = s[:,20:21,:,:] * flip(w, step*6-1)
            w7f = s[:,21:22,:,:] * flip(w, step*7-1)
            w8f = s[:,22:23,:,:] * flip(w, step*8-1)
            a = w * x + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7 + w8 * x8 + w9 * x9 + w10 * x10 + w11 * x11 + w12 * x12 + w13 * x13 + w14 * x14 + w15 * x15 + w1f * x1f + w2f * x2f  + w3f * x3f + w4f * x4f + w5f * x5f + w6f * x6f + w7f * x7f + w8f * x8f
            b = w + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10 + w11 + w12 + w13+ w14 + w15 + w1f + w2f + w3f + w4f + w5f + w6f + w7f + w8f
        else:
            print("num of symmetry is invalid.")
            exit()

        return a / b
