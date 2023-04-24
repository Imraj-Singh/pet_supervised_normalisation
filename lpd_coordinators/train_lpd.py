import os
import hydra
from omegaconf import DictConfig
import torch 
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm 
import datetime
import socket
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from src import get_lpd_model, BrainWebSupervisedTrain, LPDForwardFunction2D, LPDAdjointFunction2D, Normalisation
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cupy as xp
import cupyx.scipy.ndimage as ndi
from pyparallelproj import coincidences, resolution_models, petprojectors

@hydra.main(config_path='../configs', config_name='benchmark_lpd')
def training(config : DictConfig) -> None:
    
    model = get_lpd_model(n_iter = config.benchmark.n_iter, op = LPDForwardFunction2D, op_adj = LPDAdjointFunction2D)
    model.to(config.device)
    # train once on a combined dataset of 5, 10, 50 counts 
    print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))

    ###### SET LOGGING ######
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('./', current_time + '_' + socket.gethostname())

    writer = SummaryWriter(log_dir=log_dir)

    detector_efficiency = 1./30
    coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
        num_rings=1,
        sinogram_spatial_axis_order=coincidences.
        SinogramSpatialAxisOrder['RVP'],
        xp=xp)

    projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                                (128, 128, 1), (-127, -127, 0),
                                                (2., 2., 2.))

    res_model = resolution_models.GaussianImageBasedResolutionModel(
        (128, 128, 1), tuple(4.5 / (2.35 * x) for x in (2., 2., 2.)), xp, ndi)

    projector.image_based_resolution_model = res_model

    print("LEARNING RATE ", config.benchmark.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.benchmark.lr)

    dataset_list = []
    for noise_level in [5,10,50]:
        dataset = BrainWebSupervisedTrain(noise_level = noise_level, base_path = config.dataset.base_path)
        dataset_list.append(dataset)

    dataset = ConcatDataset(dataset_list)
    print("LENGTH OF FULL DATASET: ", len(dataset))

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    torch.random.manual_seed(42)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_dataset, batch_size=config.benchmark.batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=config.benchmark.batch_size, shuffle=False)
    
    it = iter(val_dl)
    batch_val= next(it)

    # reference, osem, measurements, contamination_factor, attn_factors
    get_norm = Normalisation(config.benchmark.normalisation)

    reference_val = batch_val[0]
    reference_val = reference_val.to(config.device)

    osem_val = batch_val[1]
    osem_val = osem_val.to(config.device)

    measurements_val = batch_val[2]
    measurements_val = measurements_val.to(config.device)

    contamination_factor_val = batch_val[3]
    contamination_factor_val = contamination_factor_val.to(config.device)

    attn_factors_val = batch_val[4]
    attn_factors_val = attn_factors_val.to(config.device)*detector_efficiency

    norm_val = get_norm(osem_val, measurements_val, contamination_factor_val)
    
    osem_grid = torchvision.utils.make_grid(osem_val, normalize=True, scale_each=True)		
    writer.add_image("osem reconstruction", osem_grid, global_step=0)

    gt_grid = torchvision.utils.make_grid(reference_val, normalize=True, scale_each=True)		
    writer.add_image("ground truth", gt_grid, global_step=0)
    
    min_loss = 100
    for epoch in range(config.benchmark.epochs):
        model.train()
        print(f"Epoch: {epoch}")
        for idx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            # reference, scale_factor, osem, norm, measurements, contamination_factor, attn_factors
            optimizer.zero_grad()

            reference = batch[0]
            reference = reference.to(config.device)

            osem = batch[1]
            osem = osem.to(config.device)

            measurements = batch[2]
            measurements = measurements.to(config.device)

            contamination_factor = batch[3]
            contamination_factor = contamination_factor.to(config.device)

            attn_factors = batch[4]
            attn_factors = attn_factors.to(config.device)*detector_efficiency

            norm = get_norm(osem, measurements, contamination_factor)

            x_pred = model(osem, measurements, projector, attn_factors, norm, contamination_factor)
            
            loss = torch.mean((x_pred - reference)**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            writer.add_scalar("train/loss_batch", loss.item(), global_step=epoch*len(train_dl) + idx)

            # SAVE SOME VALIDATION IMAGES at log_freq
            if idx % config.benchmark.log_freq == 0:
                model.eval()
                with torch.no_grad():
                    x_pred = model(osem_val, measurements_val, projector, attn_factors_val, norm_val, contamination_factor_val)
                    lpd_grid = torchvision.utils.make_grid(x_pred, normalize=True, scale_each=True)		        
                    writer.add_image("LPD reconstruction", lpd_grid, global_step=epoch*len(train_dl) + idx)
                model.train()

        # ITERATE OVER FULL VALIDATION SET
        
        val_loss = 0.
        for idx, batch in enumerate(val_dl):
            model.eval()
            with torch.no_grad():
                reference = batch[0]
                reference = reference.to(config.device)

                osem = batch[1]
                osem = osem.to(config.device)

                measurements = batch[2]
                measurements = measurements.to(config.device)

                contamination_factor = batch[3]
                contamination_factor = contamination_factor.to(config.device)

                attn_factors = batch[4]
                attn_factors = attn_factors.to(config.device)*detector_efficiency

                norm = get_norm(osem, measurements, contamination_factor)

                x_pred = model(osem, measurements, projector, attn_factors, norm, contamination_factor)

                loss = torch.mean((x_pred - reference)**2)
                val_loss += loss.item()

        val_loss = val_loss/len(val_dl)
        if val_loss < min_loss:
            print("NEW BEST VAL LOSS AT EPOCH: ", epoch)
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "model_min_val_loss.pt"))
            model.train()
        writer.add_scalar("validation/loss_average", val_loss, global_step=epoch + 1)
        torch.save(model.state_dict(), os.path.join(log_dir, "last_model.pt"))


if __name__ == '__main__':
    training()