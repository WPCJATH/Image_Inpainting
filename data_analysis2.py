'''
The python script for data analysis of this specific inpainting task.
Code are mainly duplicated from: https://github.com/nipponjo/deepfillv2-pytorch/blob/master/train.py

Author: WANG Xiangzhi
Date: 21-Dec-2023
'''

import os
import time
import torch
import torchvision as tv
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import DeepMethodBaseline.model.losses as gan_losses
import  DeepMethodBaseline.utils.misc as misc
from  DeepMethodBaseline.model.networks import Generator, Discriminator
from  DeepMethodBaseline.utils.data import ImageDataset

# Train the model
def train(config_path):
    config = misc.get_config(config_path)

    # set random seed
    if config.random_seed != False:
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        import numpy as np
        np.random.seed(config.random_seed)

    # make checkpoint folder if nonexistent
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(os.path.abspath(config.checkpoint_dir))
        os.makedirs(os.path.abspath(f"{config.checkpoint_dir}/images"))
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

    # transforms
    transforms = [T.RandomHorizontalFlip(0.5)] if config.random_horizontal_flip else None

    # dataloading
    train_dataset = ImageDataset(config.dataset_path,
                                 img_shape=config.img_shapes,
                                 random_crop=config.random_crop,
                                 scan_subdirs=config.scan_subdirs,
                                 transforms=transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available()
                          and config.use_cuda_if_available else 'cpu')
    
    # construct networks
    cnum_in = config.img_shapes[2]
    generator = Generator(cnum_in=cnum_in+2, cnum_out=cnum_in, cnum=48, return_flow=False)
    discriminator = Discriminator(cnum_in=cnum_in+1, cnum=64)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # optimizers
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.g_lr, betas=(config.g_beta1, config.g_beta2))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.d_lr, betas=(config.d_beta1, config.d_beta2))

    # losses
    if config.gan_loss == 'hinge':
        gan_loss_d, gan_loss_g = gan_losses.hinge_loss_d, gan_losses.hinge_loss_g
    elif config.gan_loss == 'ls':
        gan_loss_d, gan_loss_g = gan_losses.ls_loss_d, gan_losses.ls_loss_g
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    # resume from existing checkpoint
    last_n_iter = -1
    if config.model_restore != '':
        state_dicts = torch.load(config.model_restore)
        generator.load_state_dict(state_dicts['G'])
        if 'D' in state_dicts.keys():
            discriminator.load_state_dict(state_dicts['D'])
        if 'G_optim' in state_dicts.keys():
            g_optimizer.load_state_dict(state_dicts['G_optim'])
        if 'D_optim' in state_dicts.keys():
            d_optimizer.load_state_dict(state_dicts['D_optim'])
        if 'n_iter' in state_dicts.keys():
            last_n_iter = state_dicts['n_iter']
        print(f"Loaded models from: {config.model_restore}!")

    # start tensorboard logging
    if config.tb_logging:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.log_dir)

    device = torch.device('cuda' if torch.cuda.is_available()
                          and config.use_cuda_if_available else 'cpu')

    losses = {}

    generator.train()
    discriminator.train()

    # initialize dict for logging
    losses_log = {'d_loss':   [],
                  'g_loss':   [],
                  'ae_loss':  [],
                  'ae_loss1': [],
                  'ae_loss2': [],
                  }

    # training loop
    init_n_iter = last_n_iter + 1
    train_iter = iter(train_dataloader)
    time0 = time.time()
    for n_iter in range(init_n_iter, config.max_iters):
        # load batch of raw data
        try:
            batch_real = next(train_iter)
        except:
            train_iter = iter(train_dataloader)
            batch_real = next(train_iter)

        batch_real = batch_real.to(device, non_blocking=True)

        # create mask
        bbox = misc.random_bbox(config)
        regular_mask = misc.bbox2mask(config, bbox).to(device)
        irregular_mask = misc.brush_stroke_mask(config).to(device)
        mask = torch.logical_or(irregular_mask, regular_mask).to(torch.float32)

        # prepare input for generator
        batch_incomplete = batch_real*(1.-mask)
        ones_x = torch.ones_like(batch_incomplete)[:, 0:1].to(device)
        x = torch.cat([batch_incomplete, ones_x, ones_x*mask], axis=1)

        # generate inpainted images
        x1, x2 = generator(x, mask)
        batch_predicted = x2

        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)

        # D training steps:
        batch_real_mask = torch.cat(
            (batch_real, torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1)
        batch_filled_mask = torch.cat((batch_complete.detach(), torch.tile(
            mask, [config.batch_size, 1, 1, 1])), dim=1)

        batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask))

        d_real_gen = discriminator(batch_real_filled)
        d_real, d_gen = torch.split(d_real_gen, config.batch_size)

        d_loss = gan_loss_d(d_real, d_gen)
        losses['d_loss'] = d_loss

        # update D parameters
        d_optimizer.zero_grad()
        losses['d_loss'].backward()
        d_optimizer.step()

        # G training steps:
        losses['ae_loss1'] = config.l1_loss_alpha * \
            torch.mean((torch.abs(batch_real - x1)))
        losses['ae_loss2'] = config.l1_loss_alpha * \
            torch.mean((torch.abs(batch_real - x2)))
        losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']

        batch_gen = batch_predicted
        batch_gen = torch.cat((batch_gen, torch.tile(
            mask, [config.batch_size, 1, 1, 1])), dim=1)

        d_gen = discriminator(batch_gen)

        g_loss = gan_loss_g(d_gen)
        losses['g_loss'] = g_loss
        losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        if config.ae_loss:
            losses['g_loss'] += losses['ae_loss']

        # update G parameters
        g_optimizer.zero_grad()
        losses['g_loss'].backward()
        g_optimizer.step()


        # LOGGING
        for k in losses_log.keys():
            losses_log[k].append(losses[k].item())

        # (tensorboard) logging
        if n_iter % config.print_iter == 0:
            # measure iterations/second
            dt = time.time() - time0
            print(f"@iter: {n_iter}: {(config.print_iter/dt):.4f} it/s")
            time0 = time.time()

            # write loss terms to console
            # and tensorboard
            for k, loss_log in losses_log.items():
                loss_log_mean = sum(loss_log)/len(loss_log)
                print(f"{k}: {loss_log_mean:.4f}")
                if config.tb_logging:
                    writer.add_scalar(
                        f"losses/{k}", loss_log_mean, global_step=n_iter)                
                losses_log[k].clear()

        # save example image grids to tensorboard
        if config.tb_logging \
            and config.save_imgs_to_tb_iter \
            and n_iter % config.save_imgs_to_tb_iter == 0:
            viz_images = [misc.pt_to_image(batch_complete),
                          misc.pt_to_image(x1), misc.pt_to_image(x2)]
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2)
                        for images in viz_images]

            writer.add_image(
                "Inpainted", img_grids[0], global_step=n_iter, dataformats="CHW")
            writer.add_image(
                "Stage 1", img_grids[1], global_step=n_iter, dataformats="CHW")
            writer.add_image(
                "Stage 2", img_grids[2], global_step=n_iter, dataformats="CHW")

        # save example image grids to disk
        if config.save_imgs_to_disc_iter \
            and n_iter % config.save_imgs_to_disc_iter == 0:
            viz_images = [misc.pt_to_image(batch_real), 
                          misc.pt_to_image(batch_complete)]
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2)
                                            for images in viz_images]
            os.makedirs(f"{config.checkpoint_dir}/images", exist_ok=True)
            tv.utils.save_image(img_grids, 
            f"{config.checkpoint_dir}/images/iter_{n_iter}.png", 
            nrow=2)

        # save state dict snapshot
        if n_iter % config.save_checkpoint_iter == 0 \
            and n_iter > init_n_iter:
            misc.save_states("states.pth",
                        generator, discriminator,
                        g_optimizer, d_optimizer,
                        n_iter, config)
        # save state dict snapshot backup
        if config.save_cp_backup_iter \
            and n_iter % config.save_cp_backup_iter == 0 \
            and n_iter > init_n_iter:
            misc.save_states(f"states_{n_iter}.pth",
                        generator, discriminator,
                        g_optimizer, d_optimizer,
                        n_iter, config)





# Infer the model
def test(model_path, image_path, mask_path, out_path):
    if model_path == None:
        model_path = "model/retrained/states_25000.pth"

    generator_state_dict = torch.load(model_path)['G']

    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from DeepMethodBaseline.model.networks import Generator
    else:
        from DeepMethodBaseline.model.networks_tf import Generator  

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')

    # set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

    generator_state_dict = torch.load(model_path)['G']
    generator.load_state_dict(generator_state_dict, strict=True)

    # load image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # prepare input
    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print(f"Shape of image: {image.shape}")

    image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
    mask = (mask > 0.5).to(dtype=torch.float32,
                           device=device)  # 1.: masked 0.: unmasked

    image_masked = image * (1.-mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)  # concatenate channels

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    # complete image
    image_inpainted = image * (1.-mask) + x_stage2 * mask

    # save inpainted image
    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
    img_out = img_out.to(device='cpu', dtype=torch.uint8)
    img_out = img_out.numpy()
    img_out_ = Image.fromarray(img_out)
    img_out_.save(out_path)

    print(f"Saved output file at: {out_path}")
    return img_out


# Compute l1 loss
def compute_l1_loss(predicted, target):
    return torch.mean(torch.abs(predicted - target))

# Compute MSE loss
def compute_MSE_loss(predicted, target):
    return torch.mean(torch.square(predicted - target))

# Compute PSNR
def compute_PSNR(predicted, target):
    return 10 * torch.log10(1 / compute_MSE_loss(predicted, target))

# Compute SSIM
def compute_SSIM(predicted, target):
    return ssim(predicted.permute(1, 2, 0).cpu().numpy(), target.permute(1, 2, 0).cpu().numpy(), multichannel=True, channel_axis=2)

# Evaluate images
def evaluation(predicted, target):
    l1_loss = compute_l1_loss(predicted, target)
    MSE_loss = compute_MSE_loss(predicted, target)
    PSNR = compute_PSNR(predicted, target)
    SSIM = compute_SSIM(predicted, target)

    return l1_loss, MSE_loss, PSNR, SSIM