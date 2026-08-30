from config import settings
import copy
import logging
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from typing import Any, Callable, Dict
from IPython.display import clear_output, display

logger = logging.getLogger(__name__)

def get_kld_weight(epoch:int, cfg:Any)-> float:
    """
    calculate KLD-weight for current epoch
    Args:
        - epoch: current training epoch
        - cfg: Configuration object containing parameters like kld_weight
    Return:
        kld_weight for current epoch
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        if cfg.kld_weight_fixed:
            kldw=cfg.kld_weight
        else:
            if epoch<=30:
                kldw=cfg.kld_weight + cfg.kld_weight_max *(epoch/50)**3
            else:
                #kldw =0.1
                e1,e2=30, cfg.n_epochs
                k1,k2=cfg.kld_weight + cfg.kld_weight_max *(30/50)**3, cfg.kld_weight + 100*cfg.kld_weight_max
                m=(k2-k1)/(e2-e1)
                kldw = k1 + (epoch-e1)*m
        return kldw
    except Exception as e:
        logger.error(f"Error in get_kld_weight: {e}")
        raise


def run_one_epoch(model, dataloader, loss_fn, optimizer=None, mode: str = 'train', cfg=None, epoch:int=0):
    """
    Run one epoch for training or validation.

    Args:
        model: PyTorch model
        dataloader: PyTorch DataLoader instance
        loss_func: Loss function
        optimizer: Optimizer (required if mode='train', ignored if mode='valid'/'test')
        mode: Execution mode, either 'train' or 'valid'/'test'
        cfg: Configuration object containing parameters like cfg.device
        epoch: current epoch number

    Return:
        tuple: (total_loss_ave, recon_loss_ave, kld_loss_ave) for this epoch

    Raises:
        ValueError: If mode is 'train' but no optimizer is provided or if mode is invalid.
        Exception: Re-raises any exception caught during processing after logging.
    """
    if mode not in ['train', 'valid','test']:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'train','valid', or 'test'.")
    if mode == 'train' and optimizer is None:
        raise ValueError("An optimizer must be provided when mode='train'.")

    device = cfg.device

    try:
        model.to(device)
        is_train = (mode == 'train')
        model.train(is_train)

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kld_loss_sum = 0.0
        total_len = 0

        # Enable torch.set_grad_enabled according to mode
        with torch.set_grad_enabled(is_train):
            for batch_i, (real_images, _) in enumerate(dataloader, start=1):
                real_images = real_images.to(cfg.device)

                if is_train:
                    optimizer.zero_grad()

                recon_images, mu, logvar = model(real_images)

                # if cfg.kld_weight_fixed:
                #     kldw=cfg.kld_weight
                # else:
                #     kldw=cfg.kld_weight + cfg.kld_weight_max *(epoch/cfg.n_epochs)**3
                #     # if epoch <=50:
                #     #     kldw=cfg.kld_weight + cfg.kld_weight_max *(epoch/50)**3
                #     # else:
                #     #     kldw=cfg.kld_weight + cfg.kld_weight_max
                kldw= get_kld_weight(epoch, cfg)

                loss_dict = loss_fn(recon_x=recon_images, x=real_images, mu=mu, logvar=logvar,
                                    kld_weight=kldw, reduction=cfg.reduction, rec_loss_type = cfg.rec_loss_type)

                if is_train:
                    loss_dict["total_loss"].backward()
                    optimizer.step()

                # Accumulate loss
                batch_size = real_images.size(0)
                if cfg.reduction=='mean':
                    scale = batch_size
                else:
                    scale = 1

                total_loss_sum += loss_dict["total_loss"].detach() * scale
                recon_loss_sum += loss_dict["recon_loss"].detach() * scale
                kld_loss_sum += loss_dict["kld_loss"].detach() * scale
                total_len += batch_size

        total_loss_ave = total_loss_sum.item() / total_len
        recon_loss_ave = recon_loss_sum.item() / total_len
        kld_loss_ave = kld_loss_sum.item() / total_len

        return total_loss_ave, recon_loss_ave, kld_loss_ave, kldw

    except Exception as e:
        logger.error(f"Error in run_one_epoch (mode={mode}): {e}")
        raise


def train_model(model, dataloaders, loss_fn, optimizer, scheduler, cfg, jupyter_vis=True):
    """
    train the model
    Args:
        model: pytorch model
        dataloaders: a dict of train/valid/tes data-loaders
        loss_fn: loss function
        optimizer: optimizer
        scheduler: scheduler to decrease learning-rate as training progress
        device: device type cpu or cuda/gpu
        jupyter_vis: a boolean flag to enable/disable real-time visualization of training loss/accuracy per epoch
    Return:
        model, a dataframe contains history of loss/accuracy for both train and validation, and the fig
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        # Ensure dataloaders is a dictionary with the required keys
        assert isinstance(dataloaders, dict), "dataloaders must be a dictionary."
        assert 'train' in dataloaders and 'valid' in dataloaders, "Missing 'train' or 'valid' in dataloaders."

        model_name = cfg.model_name #model_name: a name for model for tracking/visualization/saving

        best_model_param_path = cfg.best_model_param_path #settings.OUT_DIR/f'best_model_param_{model_name}.pth'
        last_model_param_path = cfg.last_model_param_path #settings.OUT_DIR/f'last_model_param_{model_name}.pth'
        training_history_path = cfg.training_history_path #settings.OUT_DIR/f'training_history_{model_name}.csv'
        generated_samples_path = cfg.generated_samples_path #settings.OUT_DIR/f'generated_samples_{model_name}.pkl'
        fixed_z = torch.randn(cfg.num_image_to_display_during_training, cfg.latent_dim).to(cfg.device)
        generated_images_list=[]


        history_list = []
        #best_val_acc = 0
        valid_loss_min = None
        valid_loss_min_epoch = 0

        for epoch in range(1,1+cfg.n_epochs):
            epoch_metrics = {'model': model_name}
            epoch_metrics['epoch'] = epoch
            epoch_pct = round(100*epoch/cfg.n_epochs,2)
            epoch_metrics['progress%'] = epoch_pct

            train_loss, train_recon_loss, train_kld_loss, kldw =run_one_epoch(
                model, dataloaders['train'], loss_fn, optimizer, 'train', cfg,epoch)
            epoch_metrics['train_loss'] = train_loss
            epoch_metrics['train_recon_loss'] = train_recon_loss
            epoch_metrics['train_kld_loss'] = train_kld_loss
            epoch_metrics['train_kld_weight'] = kldw

            valid_loss, valid_recon_loss, valid_kld_loss, kldw =run_one_epoch(
                model, dataloaders['valid'], loss_fn, None, 'valid', cfg,epoch)
            epoch_metrics['valid_loss'] = valid_loss
            epoch_metrics['valid_recon_loss'] = valid_recon_loss
            epoch_metrics['valid_kld_loss'] = valid_kld_loss
            epoch_metrics['valid_kld_weight'] = kldw

            if valid_loss_min is None or ((valid_loss_min - valid_loss) / valid_loss_min > 0.0001):
                valid_loss_min = valid_loss
                valid_loss_min_epoch = epoch
                torch.save(model.state_dict(), best_model_param_path)

            # Update history
            history_list.append(epoch_metrics)
            history_df = pd.DataFrame(history_list)
            scheduler.step() # update in each epoch we could also update in each batch!
            # Save final history to CSV
            history_df.to_csv(training_history_path, index=False)
            #update checkpoint if needed!
            model.eval()
            with torch.no_grad():
                generated_images = model.decoder(fixed_z)
                generated_images_list.append(generated_images.cpu())
            model.train()


            if jupyter_vis:# Jupyter Visualization
                #clear_output(wait=True)
                #display(history_df.tail(min([10, cfg.n_epochs])))

                # grid = torchvision.utils.make_grid(generated_images, nrow=4, padding=2, normalize=True)
                # fig, axs = plt.subplots(2, 2, figsize=(9, 9))
                # axs[0,0].imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
                # axs[0,0].axis("off")
                # axs[0,0].set_title("Generated Images")
                # history_df.plot(x='epoch', y=['train_loss', 'valid_loss'], ax=axs[0,1], grid=True, marker='o',
                #                 title=f'{model_name}-Total-Loss', ylabel='Total-Loss')
                # history_df.plot(x='epoch', y=['train_recon_loss', 'valid_recon_loss'], ax=axs[1,0], grid=True, marker='d',
                #                                 title=f'{model_name}-Recon-Loss', ylabel='Recon-Loss')
                # history_df.plot(x='epoch', y=['train_kld_loss', 'valid_kld_loss'], ax=axs[1,1],  grid=True, marker='s',
                #                 title=f'{model_name}-KLD-Loss', ylabel='KLD-Loss')
                # plt.tight_layout()
                # plt.show()
                #display(history_df.tail(1))
                if epoch==1 or epoch==cfg.n_epochs or epoch%cfg.show_every==0:
                    #print(f'{epoch_metrics}')
                    display(history_df.tail(1))
                    grid = torchvision.utils.make_grid(generated_images, nrow=4, padding=2, normalize=True, value_range=(0, 1))
                    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
                    axs[0].imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
                    axs[0].axis("off")
                    axs[0].set_title(f"Generated Images (Epoch-{epoch})")
                    history_df.plot(x='epoch', y=['train_loss', 'valid_loss'], ax=axs[1], grid=True, marker='o',
                                    title=f'{model_name}-Best-Valid at Epoch-{valid_loss_min_epoch}', ylabel='Total-Loss')
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f'Epoch:{epoch:02d}, Progress:{epoch_pct:.1f}%, Total-Loss: Train= {train_loss:.6f}, Valid= {valid_loss:.6f}')

                #plt.close(fig)
            else:
                print(f'{epoch_metrics}')
                fig = None
        #fig.savefig(settings.IMG_DIR/f'training-{model_name}.png', dpi=600)
        with open(generated_samples_path, "wb") as f:
            pkl.dump(generated_images_list, f)
        print(f'Best Validation Total Loss ={valid_loss_min:.6f} at Epoch {valid_loss_min_epoch}')
        torch.save(model.state_dict(), last_model_param_path)
        return model, history_df, fig, valid_loss_min_epoch
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise


def eval_model_on_dataset(model, dataloader, loss_fn, cfg=None, epoch=1) :
    """
    Run one epoch for evaluation of dataset (dataloader)

    Args:
        model: PyTorch model
        dataloader: PyTorch DataLoader instance
        loss_func: Loss function
        cfg: Configuration object containing parameters like cfg.device
        epoch: epoch number, valid_loss_min_epoch (for best-model) or max_epoch (for last model)
    Return:
        tuple: (total_loss_ave, recon_loss_ave, kld_loss_ave) for this epoch

    Raises:
        ValueError: If mode is 'train' but no optimizer is provided or if mode is invalid.
        Exception: Re-raises any exception caught during processing after logging.
    """
    device = cfg.device

    try:
        model.to(device)
        model.train(False) # or model.eval()

        total_loss_list = []
        recon_loss_list = []
        kld_loss_list = []

        # Enable torch.set_grad_enabled according to mode
        with torch.set_grad_enabled(False):
            for batch_i, (real_images, _) in enumerate(dataloader, start=1):
                real_images = real_images.to(cfg.device)

                recon_images, mu, logvar = model(real_images)

                if cfg.kld_weight_fixed:
                    kldw=cfg.kld_weight
                else:
                    kldw=cfg.kld_weight + cfg.kld_weight_max *(epoch/cfg.n_epochs)**3

                loss_dict = loss_fn(recon_x=recon_images, x=real_images, mu=mu, logvar=logvar,
                                    kld_weight=kldw, reduction='none', rec_loss_type = cfg.rec_loss_type)

                #batch_size = real_images.size(0)
                #total_len += batch_size

                # total_loss_list.append(loss_dict["total_loss"].detach().cpu().item())
                # recon_loss_list.append(loss_dict["recon_loss"].detach().cpu().item())
                # kld_loss_list.append(loss_dict["kld_loss"].detach().cpu().item())

                total_loss_list.append(loss_dict["total_loss"].detach().cpu())
                recon_loss_list.append(loss_dict["recon_loss"].detach().cpu())
                kld_loss_list.append(loss_dict["kld_loss"].detach().cpu())

        # Concatenate all batches and convert directly to 1D NumPy arrays
        total_losses = torch.cat(total_loss_list).numpy()
        recon_losses = torch.cat(recon_loss_list).numpy()
        kld_losses = torch.cat(kld_loss_list).numpy()

        #return np.array(total_loss_list), np.array(recon_loss_list), np.array(kld_loss_list)
        return total_losses, recon_losses, kld_losses

    except Exception as e:
        logger.error(f"Error in eval_model_on_dataset: {e}")
        raise
