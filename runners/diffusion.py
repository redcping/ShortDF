import os
import logging
import time
import glob
import copy
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torchvision.utils import make_grid, save_image
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from score.both import get_inception_and_fid_score

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
      
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            drop_last = True
        )
        model = Model(config)
        graph_model = Model(config)
      
        ckpt_fn = f'./pretrained_models/{args.config[:-4]}.ckpt'
        
        if ckpt_fn!='':
            ckpt = torch.load(ckpt_fn, map_location=self.device)
            try:
                model.load_state_dict(ckpt)
                print(f'{args.config[:-4]}.ckpt')
            except:
                model.load_state_dict(ckpt[-1])
                print(ckpt_fn)
                pass
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        
        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        graph_model = ema_helper.ema_copy(model).eval()
        total_steps = self.config.training.n_epochs * len(train_loader)
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0  
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x) 
                b = self.betas

                #antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
              

                ema_model = ema_helper.ema_copy(model).eval()
                loss_dict = loss_registry[config.model.type](model, x, t, e, b, 
                                                             noise_weight=config.training.noise_weight, 
                                                             relax_weight=config.training.relax_weight, 
                                                             graph_model=graph_model,
                                                             ema_model=ema_model)
                loss = loss_dict['total_loss']
                optimizer.zero_grad()
                loss.backward()
                
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step==1 or step % 200 ==0:
                    loss_info = f'loss:{loss.item():.3f} '
                    for k, v in loss_dict.items():
                        tb_logger.add_scalar(k, v, global_step=step)
                        loss_info=loss_info+f"{k}:{v.item():.3f} "
                    logging.info(
                        f"step: {step}, {loss_info}, data time: {data_time / (i+1):.4f}"
                    )


                if step==1 or step % 500 ==0:
                    with torch.no_grad():
                        k = np.random.randint(10)+1# if np.random.rand()>0.5 else 0
                        k = 10 if step==1 else k
                        nrow = int(np.sqrt(x.size(0)))
                        x_T = torch.randn_like(x[:nrow**2])
                        ema_model = ema_helper.ema_copy(model).eval()
                        #ema_model = graph_model
                        x_0 = self.sample_image(x_T, ema_model, last=True, sample_timesteps=k)
                        x_0 = inverse_data_transform(config, x_0)
                        grid = make_grid(x_0,nrow=nrow)
                        path = os.path.join(self.args.log_path, 'sample', '{}_{}.png'.format(step,k))
                        os.makedirs(os.path.join(self.args.log_path, 'sample'),exist_ok=True)
                        save_image(grid, path)
                        tb_logger.add_image('sample', grid, step)
                        del ema_model

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    if step%10000==0:
                        torch.save(
                            states,
                            os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                        )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()


                    
        

  
    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            print(os.path.join(self.args.log_path, "ckpt.pth"),self.config.model.ema)
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            
            model.load_state_dict(states[-1], strict=True)
            print('ema:',os.path.join(self.args.log_path, "ckpt.pth"), self.config.model.ema)
           
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            # Place the downloaded model into the pretrained_models folder and rename it according to the dataset name.
            # For example, rename the one for CIFAR-10 to cifar10.ckpt
            
            print('use pretrained model')
            ckpt = f'./pretrained_models/{self.args.config[:-4]}.ckpt'
            print("Loading checkpoint {}".format(ckpt))
            try:
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
            except:
                model.load_state_dict(torch.load(ckpt, map_location=self.device)[-1])
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")




    def sample_fid(self, model):
      
        config = self.config
        img_id = 0
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        print(total_n_samples)
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        image_list=[]
        
        with torch.no_grad():
            for _ in tqdm.tqdm(range(n_rounds), desc="Generating image samples for FID evaluation."):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x).cpu()
                image_list.append(x)

        images = torch.cat(image_list, dim=0).numpy()
 
        try:
            save_folder = os.path.join(self.args.log_path,'images_samples_test')
            os.makedirs(save_folder,exist_ok=True)
            pretrained_str='ddim' if self.args.use_pretrained else 'ours'
            save_image(
                torch.tensor(images[:64]),
                os.path.join(save_folder, f'samples_ema_{self.args.timesteps}_{self.args.eta}_{pretrained_str}.png'),
                nrow=8)
        except:
            pass
       
        fid_cache = f'stats/{self.args.config[:-4]}_train_fid_stats.npz'
       
        (IS, IS_std), FID = get_inception_and_fid_score(
        images, fid_cache, num_images=total_n_samples,
        use_torch=False, verbose=True)
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        

                
    def sample_sequence(self, model):
        config = self.config
        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )


        #NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
           _, x = self.sample_image(x, model, last=False)
        

        x = [inverse_data_transform(config, y) for y in x]
        pretrained_str='pretrained_' if self.args.use_pretrained else ''
        save_folder = os.path.join(self.args.log_path,pretrained_str+f'{self.args.config[:-4]}_{self.args.timesteps}_consis_images_samples_eta0')
        os.makedirs(save_folder,exist_ok=True)
        # seq=[t+1 for t in seq] + [0]
        # print(len(seq),len(x),seq)
        print(len(x))
        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(save_folder, f"{j}_{i+1}.png")
                )
                

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True, sample_timesteps=0):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        sample_timesteps =  self.args.timesteps if sample_timesteps==0 else sample_timesteps
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                #skip = self.num_timesteps // sample_timesteps
                #seq = range(0, self.num_timesteps, skip)
                seq = [int(x) for x in np.linspace(0, self.num_timesteps-1, sample_timesteps)]
                seq = seq if sample_timesteps>1 else [self.num_timesteps-1]
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), sample_timesteps
                    )
                    ** 2
                )
                seq =[int(s) for s in list(seq)]
                if len(seq)<6:
                    s1=[700]
                    s2=[200,700]
                    s3=[0,200,700]
                    s4=[0,40, 200,700]
                    s5=[0,100,200,700,800]
                    ss=[s1,s2,s3,s4,s5]
                    seq = ss[len(seq)-1]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            # print(seq)
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                #skip = self.num_timesteps // sample_timesteps
                #seq = range(0, self.num_timesteps, skip)
                seq = [int(x) for x in np.linspace(0, self.num_timesteps-1, sample_timesteps)]
                seq = seq if sample_timesteps>1 else [800]
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), sample_timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x
       

    def test(self):
        pass
