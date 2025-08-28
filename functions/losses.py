import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    noise_loss =  (e - output).square().sum(dim=(1, 2, 3))

    loss_dict={}
    loss_dict['total_loss'] = noise_loss.mean()
    return loss_dict



def shordf_relax_loss(model,
                  x0: torch.Tensor,
                  t: torch.LongTensor,
                  e: torch.Tensor,
                  b: torch.Tensor, 
                  noise_weight=1.0,
                  relax_weight=1.0,
                  ode_model=None, ema_model=None):
    
    def get_x0(xi, ei, ai):
        return torch.clamp((xi - ei * (1 - ai).sqrt()) / ai.sqrt(),-1,1)

    def ddim_i2j(x0_i, ai, aj, ei, eta=0):
        sigma = eta * ((1 - ai / aj) * (1 - aj) / (1 - ai)).sqrt()
        return aj.sqrt() * x0_i + (1 - aj - sigma**2).sqrt() * ei  + sigma * torch.randn_like(ei)

    def calc_dist(xi, ai, i, f, return_all=False):
        eps = f(xi, torch.clamp(i-1, 0).float()) * (i>0).view(-1,1,1,1).float()
        x0_i = get_x0(xi, eps, ai)
        return [x0_i, eps, x0_i - x0] if return_all else x0_i - x0
    
    alphas_bar = (1-b).cumprod(dim=0)
    alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)
    t = torch.clamp(t, 1)
    m = (torch.rand_like(t.float())*t).to(t)
    
    at = alphas_bar.index_select(0, t).view(-1, 1, 1, 1)
    am = alphas_bar_prev.index_select(0, m).view(-1, 1, 1, 1)

    xt = at.sqrt() * x0 + (1-at).sqrt() * e
    loss_dict={}
    eta=0
    margin = 1e-4
    with torch.no_grad():
        assert ema_model is not None,'pls give ema_model'
        ema_model = ema_model if ema_model is not None else model
        ode_et = ode_model(xt, t.float())
        ode_x0_t = get_x0(xt, ode_et, at)

        xm = ddim_i2j(x0, at, am, ode_et, eta)
        dm = calc_dist(xm, am, m, ode_model)
       
        xt2m = ddim_i2j(ode_x0_t, at, am, ode_et, eta)
        dt2m = calc_dist(xt2m, am, m, ode_model, False)

        edge = dt2m.abs() - dm.abs()

        dist_m = calc_dist(xm, am, m, ema_model, False).abs()
        dist_update = F.relu(dist_m + edge) + margin

    et = model(xt, t.float())
    loss_dict['noise_loss'] = noise_weight * (e-et).square().mean()
    
    dist_t = (get_x0(xt, et, at) - x0).abs()
    
    relax_mask = dist_t > dist_update
    relax_loss = relax_mask.float() * (dist_t - dist_update)
    loss_dict['relax_loss'] = relax_weight * relax_loss.mean()

    # mask_0  = (t<10).view(-1,1,1,1)
    # _,c,h,w = x0.size()
    # dist_0 = (dist_t.square() * mask_0).sum()/(mask_0.sum() * c * h * w + 1)
    # loss_dict['dist_0'] = dist_0.mean()
    
    total_loss = 0
    for k,v in loss_dict.items():
        total_loss = total_loss + v
    loss_dict['total_loss'] = total_loss

    return loss_dict









loss_registry = {
    'simple': noise_estimation_loss,
    'shordf_relax_loss':shordf_relax_loss,
   
}



