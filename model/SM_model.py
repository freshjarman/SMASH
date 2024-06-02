import math
import torch
from torch import nn
import torch.nn.functional as F


def normalize_to_neg_one_to_one(img):
    img_new = img * 2 - 1
    if img.size(-1) == 4:
        img_new[:,:,1]=img[:,:,1]
    return img_new

def unnormalize_to_zero_to_one(img):
    img_new = (img + 1) * 0.5
    if img.size(-1) == 4:
        img_new[:,:,1]=img[:,:,1]
    return img_new

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class ScoreMatch_module(nn.Module):
    def __init__(self, dim, num_units=64,self_condition = False,condition=True,cond_dim=0, num_types=1):

        super(ScoreMatch_module, self).__init__()
        self.channels = 1
        self.self_condition = self_condition
        self.condition = condition
        self.cond_dim=cond_dim

        sinu_pos_emb = SinusoidalPosEmb(num_units)
        fourier_dim = num_units
        self.num_types = num_types

        time_dim = num_units

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.linears_spatial = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
            ]
        )

        self.linears_temporal = nn.ModuleList(
            [
                nn.Linear(1, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
            ]
        )

        self.output_intensity = nn.Sequential(
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_types),
                nn.Softplus(beta=1)

        )
        self.output_score = nn.Sequential(
                nn.Linear(num_units * 2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2)

        )

        self.linear_t = nn.Sequential(
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2)
        )

        self.linear_s = nn.Sequential(
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2)
        )


        self.cond_all = nn.Sequential(
                nn.Linear(cond_dim * 3 if num_types==1 else cond_dim * 4, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units)
        )

        self.cond_temporal = nn.ModuleList(
            [
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units)
            ]
        )

        self.cond_spatial = nn.ModuleList(
            [
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units)
            ]
        )

        self.cond_joint = nn.ModuleList(
            [
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units)
            ]
        )

    def get_intensity(self, t, cond):
        x_temporal = t


        hidden_dim = self.cond_dim

        cond_temporal, cond_spatial, cond_joint, cond_mark = cond[:,:,:hidden_dim], cond[:,:,hidden_dim:2*hidden_dim], cond[:,:,(2*hidden_dim):(3*hidden_dim)],cond[:,:,3*hidden_dim:] 

        cond = self.cond_all(cond)


        for idx in range(3):
            x_temporal = self.linears_temporal[2 * idx](x_temporal)

            cond_joint_emb = self.cond_joint[idx](cond_joint)

            cond_temporal_emb = self.cond_temporal[idx]((cond_temporal+cond_mark) if self.num_types>1 else cond_temporal)
            
            x_temporal += cond_joint_emb + cond_temporal_emb
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_temporal = self.linears_temporal[-1](x_temporal)

        pred = self.output_intensity(x_temporal)
        return pred

    def get_score_loc(self, x, cond):
        x_spatial, x_temporal = x[:,:,1:], x[:,:,:1]


        hidden_dim = self.cond_dim

        cond_temporal, cond_spatial, cond_joint, cond_mark = cond[:,:,:hidden_dim], cond[:,:,hidden_dim:2*hidden_dim], cond[:,:,2*hidden_dim:3*hidden_dim],cond[:,:,3*hidden_dim:] 

        cond = self.cond_all(cond)

        alpha_s = F.softmax(self.linear_s(cond), dim=-1).squeeze(dim=1).unsqueeze(dim=2)
        alpha_t = F.softmax(self.linear_t(cond), dim=-1).squeeze(dim=1).unsqueeze(dim=2)


        for idx in range(3):
            x_spatial = self.linears_spatial[2 * idx](x_spatial)
            x_temporal = self.linears_temporal[2 * idx](x_temporal)

            cond_joint_emb = self.cond_joint[idx](cond_joint)
            cond_temporal_emb = self.cond_temporal[idx]((cond_temporal+cond_mark) if self.num_types>1 else cond_temporal)
            cond_spatial_emb = self.cond_spatial[idx](cond_spatial)

            x_spatial += cond_joint_emb + cond_spatial_emb
            x_temporal += cond_joint_emb + cond_temporal_emb

            x_spatial = self.linears_spatial[2 * idx + 1](x_spatial)
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_spatial = self.linears_spatial[-1](x_spatial)
        x_temporal = self.linears_temporal[-1](x_temporal)

        x_output_t = x_temporal * alpha_t[:,:1,:] + x_spatial * alpha_t[:,1:2,:]
        x_output_s = x_temporal * alpha_s[:,:1,:] + x_spatial * alpha_s[:,1:2,:]

        pred = self.output_score(torch.cat((x_output_t, x_output_s), dim=-1))
        return pred

    def get_score(self, x, cond = None, sample=True):
        t = torch.autograd.Variable(x[:,:,:1], requires_grad=True)

        intensity = self.get_intensity(t, cond)
        intensity_log = (intensity+1e-10).log()

        intensity_grad_t = torch.autograd.grad(intensity_log.sum(), t, retain_graph=True, create_graph=sample)[0]
        score_t = intensity_grad_t - intensity
        score_loc = self.get_score_loc(x, cond)

        return torch.cat((score_t,score_loc),-1)

    def get_score_mark(self, x, mark, cond = None, sample=True):

        t = torch.autograd.Variable(x[:,:,:1], requires_grad=True)

        intensity = self.get_intensity(t, cond)
        # intensity_total = intensity.sum(-1)
        mark_onehot = F.one_hot(mark.long(), num_classes=self.num_types) # batch*(len-1)*num_samples*num_types
        # print(mark_onehot.size(),intensity.size())
        intensity_mark = (mark_onehot*intensity).sum(-1)
        intensity_mark_log = (intensity_mark+1e-10).log()


        intensity_grad_t = torch.autograd.grad(intensity_mark_log.sum(), t, retain_graph=True, create_graph=sample)[0]
        score_t = intensity_grad_t - intensity.sum(-1,keepdim=True)
        score_loc = self.get_score_loc(x, cond)
        score_mark = intensity / (intensity.sum(-1).unsqueeze(-1)+1e-10)


        return torch.cat((score_t,score_loc),-1), score_mark


class SMASH(nn.Module):
    def __init__(
        self,
        model,
        sigma,
        seq_length,
        num_noise = 50,
        sampling_timesteps = 500,
        langevin_step = 0.05,
        n_samples = 300,
        sampling_method = 'normal',
        num_types = 1,
        loss_lambda =1,
        loss_lambda2 =1,
        smooth = 0.0
    ):
        super(SMASH, self).__init__()
        self.model = model
        self.channels = n_samples
        self.num_noise = num_noise
        self.self_condition = self.model.self_condition
        self.is_marked = num_types > 1
        self.num_types = num_types
        self.loss_lambda = loss_lambda
        self.loss_lambda2 = torch.tensor([1.,loss_lambda2, loss_lambda2]).cuda()
        self.smooth = smooth

        self.seq_length = seq_length
        self.sampling_timesteps = sampling_timesteps
        self.sigma = torch.tensor([sigma[0],sigma[1],sigma[1]]).cuda()
        self.langevin_step = langevin_step
        self.n_samples = n_samples
        self.sampling_method = sampling_method

    def sample_from_last(self, batch_size = 16, step = 100, is_last = False, cond=None, last_sample = None):
        seq_length, channels = 3, self.channels
        shape = (batch_size, channels, seq_length)
        e = self.langevin_step
        n_samples = self.n_samples

        if not self.is_marked:
            if last_sample is not None:
                x = normalize_to_neg_one_to_one(last_sample[0])
            else:
                x = torch.randn([*shape], device=cond.device)

            sqrt_e = math.sqrt(e)

            if self.sampling_method == 'normal':
                for _ in range(step):
                    z = torch.randn_like(x)
                    score = self.model.get_score(x, cond, False)
                    x = x + 0.5 * e * score.detach() + sqrt_e * z
                    x.clamp_(-1., 1.)
        
            if is_last:
                score = self.model.get_score(x, cond, False)
                x_final = x + self.sigma**2 * score.detach()
            else:
                x_final = x
            x.clamp_(-1., 1.)


            x_final.required_grads=False
        
            img = unnormalize_to_zero_to_one(x_final)
            return (img.detach(),None)
        else:
            if last_sample is not None:
                x, score_mark = last_sample
                x = normalize_to_neg_one_to_one(x)
                mark = torch.multinomial(score_mark.reshape(-1, self.num_types)+1e-10,1, replacement=False).reshape(batch_size,n_samples) # batch*len-1*num_samples  
            
            else:
                x = 0.5*torch.randn([*shape], device=cond.device)
                mark = torch.multinomial(torch.ones(self.num_types).cuda(),batch_size*n_samples, replacement=True).reshape(batch_size,n_samples).cuda() # batch*len-1*num_samples  
    
# torch.ones(self.num_types).cuda()
            sqrt_e = math.sqrt(e)

            if self.sampling_method == 'normal':
                for s in range(step):
                    z = torch.randn_like(x)
                    score, score_mark = self.model.get_score_mark(x, mark, cond, False)
                    x = x + 0.5 * e * score.detach() + sqrt_e * z
                    x.clamp_(-1., 1.)
                    mark = torch.multinomial(score_mark.detach().reshape(-1, self.num_types)+1e-10,1, replacement=False).reshape(batch_size,n_samples) # batch*len-1*num_samples  
                    # if s >10:
                    #     return 0
        
            if is_last:
                score, _ = self.model.get_score_mark(x, mark, cond, False)
                x_final = x + self.sigma**2 * score.detach()
                _, score_mark = self.model.get_score_mark(x_final, mark, cond, False)
                mark = torch.multinomial(score_mark.detach().reshape(-1, self.num_types)+1e-10,1, replacement=False).reshape(batch_size,n_samples) # batch*len-1*num_samples  
                for s in range(200):
                    z = torch.randn_like(x)
                    score, score_mark = self.model.get_score_mark(x_final, mark, cond, False)
                    x_final[:,:,1:] = x_final[:,:,1:] + 0.5 * e * score.detach()[:,:,1:] + sqrt_e * z[:,:,1:]
            else:
                x_final = x
            x_final.clamp_(-1., 1.)


            x_final.required_grads=False
        
            img = unnormalize_to_zero_to_one(x_final)
            return (img.detach(), score_mark.detach())


    def p_losses(self, x_start, noise = None, cond=None):
        noise = default(noise, lambda: torch.randn_like(x_start.repeat(1,self.num_noise,1)))

        # noise sample
        x = x_start + self.sigma * noise

        score = self.model.get_score(x, cond)

        loss = self.get_obj_denoise(x_start, x, score)

        return loss.mean()
    
    def p_losses_mark(self, x_start, noise = None, cond=None):
        x_mark = x_start[:,:,1]
        x_start = torch.cat((x_start[:,:,:1], x_start[:,:,2:]),-1)
        noise = default(noise, lambda: torch.randn_like(x_start.repeat(1,self.num_noise,1)))

        # noise sample
        x = x_start + self.sigma * noise

        score, score_mark = self.model.get_score_mark(x, x_mark-1, cond)

        loss = self.get_obj_denoise(x_start, x, score)
        loss *= self.loss_lambda2
        loss_mark = self.get_obj_mark(x_mark, score_mark, smooth=self.smooth)

        return loss.mean() + self.loss_lambda * loss_mark.mean()


    def get_obj_denoise(self, x_start, x, score):
        target = (x_start - x)/self.sigma**2
        # print('t',target[0][0])
        # print('s',score[0][0])
        obj = 0.5 * (score - target)**2
        obj *= self.sigma**2
        # print(obj.mean(0).mean(0))
        return obj

    def get_obj_mark(self, x_mark, score_mark, smooth=0.0):
        truth = x_mark - 1
        one_hot = F.one_hot(truth.long(), num_classes=self.num_types).float()
        one_hot = one_hot * (1 - smooth) + (1 - one_hot) * smooth / self.num_types
        log_prb = (score_mark+1e-10).log()
        obj = -(one_hot * log_prb).sum(dim=-1)
        return obj

    def forward(self, img, cond, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        img = normalize_to_neg_one_to_one(img)
        
        if not self.is_marked:
            loss = self.p_losses(img, cond=cond,  *args, **kwargs)
        else:
            loss = self.p_losses_mark(img, cond=cond,  *args, **kwargs)
        
        return loss



class Model_all(nn.Module):
    def __init__(self, transformer, decoder):
        super(Model_all, self).__init__()
        self.transformer = transformer
        self.decoder = decoder
