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
from src import get_unet_model, Normalisation, BrainWebSupervisedTrain
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@hydra.main(config_path='../configs', config_name='benchmark_unet')
def training(config : DictConfig) -> None:

    model = get_unet_model(in_ch=config.benchmark.in_ch, 
                           out_ch=config.benchmark.out_ch, 
                           scales=config.benchmark.scales, 
                           skip=config.benchmark.skip,
                           channels=config.benchmark.channels, 
                           use_sigmoid=config.benchmark.use_sigmoid,
                           use_norm=config.benchmark.use_norm)
    
    model.to(config.device)
    # train once on a combined dataset of 5, 50, 500 counts 
    print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))

    ###### SET LOGGING ######
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('./', current_time + '_' + socket.gethostname())

    writer = SummaryWriter(log_dir=log_dir)

    print("LEARNING RATE ", config.benchmark.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.benchmark.lr)

    dataset_list = []
    for noise_level in config.benchmark.train_on_noise_level:
        dataset = BrainWebSupervisedTrain(noise_level=noise_level, base_path = config.dataset.base_path)
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

            norm = get_norm(osem, measurements, contamination_factor)

            x_pred = model(osem, norm)
            
            loss = torch.mean((x_pred - reference)**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            writer.add_scalar("train/loss_batch", loss.item(), global_step=epoch*len(train_dl) + idx)

            # SAVE SOME VALIDATION IMAGES at log_freq
            if idx % config.benchmark.log_freq == 0:
                model.eval()
                with torch.no_grad():
                    x_pred = model(osem_val, norm_val)
                    unet_grid = torchvision.utils.make_grid(x_pred, normalize=True, scale_each=True)		        
                    writer.add_image("UNet reconstruction", unet_grid, global_step=epoch*len(train_dl) + idx)
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

                norm = get_norm(osem, measurements, contamination_factor)

                x_pred = model(osem, norm)

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