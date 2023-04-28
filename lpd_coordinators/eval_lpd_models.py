import torch 
from torch.utils.data import DataLoader
from glob import glob
import torch
import numpy as np 
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio
import omegaconf
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from src import get_lpd_model, LPDForwardFunction2D, LPDAdjointFunction2D, Normalisation
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pyparallelproj import coincidences, resolution_models, petprojectors
import cupy as xp
import cupyx.scipy.ndimage as ndi

detector_efficiency = 1./30
coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(num_rings=1,
    sinogram_spatial_axis_order=coincidences.SinogramSpatialAxisOrder['RVP'],xp=xp)

projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                            (128, 128, 1), (-127, -127, 0),
                                            (2., 2., 2.))

res_model = resolution_models.GaussianImageBasedResolutionModel(
    (128, 128, 1), tuple(4.5 / (2.35 * x) for x in (2., 2., 2.)), xp, ndi)

projector.image_based_resolution_model = res_model

pnll = torch.nn.PoissonNLLLoss(log_input=False,reduction='sum')


models = ["none", "data_corrected_mean", "data_mean", "osem_mean", "osem_max"]
model_base_path = "E:/projects/pet_supervised_normalisation/lpd_coordinators/LPD/"
dataset_path = "E:/projects/pet_score_model/src/brainweb_2d/"
save_idxs = [10,20,30,40,50,60]
for model_type in models:
    model_path = model_base_path + model_type + "/"
    config = omegaconf.OmegaConf.load(model_path + ".hydra/config.yaml")
    model = get_lpd_model(n_iter = config.benchmark.n_iter, op = LPDForwardFunction2D, op_adj = LPDAdjointFunction2D)
    model.load_state_dict(torch.load(sorted(glob(model_path+"/*/model_min_val_loss.pt"))[-1]))
    model.eval()
    model.to(config.device)
    noisy_paths = sorted(glob(dataset_path + "noisy/*test*.pt"))
    clean_path = dataset_path + "clean/"
    noisy = [noise.rsplit("\\")[-1][:-3] for noise in noisy_paths]
    qm_dicts = {}
    get_normalisation = Normalisation(config.benchmark.normalisation)
    print(f"Model type: {model_type}")
    for idx in range(len(noisy)):
        noise_level = noisy[idx].rsplit("_",2)[-1]
        if "tumour" in noisy[idx]:
            tumour = True
            clean_dataset = torch.load(clean_path + "clean_test_tumour.pt")
            std_s = []
            crc_s_r_t = []
        else:
            tumour = False
            clean_dataset = torch.load(clean_path + "clean_test.pt")

        noisy_path = noisy_paths[idx]
        noisy_dataset = torch.load(noisy_path)
        n_samples, n_realisations = noisy_dataset["measurements"].shape[:2]
        x_preds_save = []
        x_ref_save = []
        ssim_s_r = []
        psnr_s_r = []
        pnll_s_r = []
        kldiv_s_r = []
        print(f"Processing {noisy[idx]}")
        for s in tqdm(range(n_samples)):
            reference = clean_dataset["reference"][s:s+1,0:1,...].to(config.device) * noisy_dataset["scale_factor"][s:s+1,0:1,...].to(config.device)
            if tumour:
                tumour_rois = clean_dataset["tumour_rois"][s:s+1,...].to(config.device)
                background = clean_dataset["background"][s:s+1,...].to(config.device)
                crc_r_t = []
                b_bar_r = []
            ssim_r = []
            psnr_r = []
            pnll_r = []
            kldiv_r = []

            for r in range(n_realisations):
                osem = noisy_dataset["osem"][s:s+1,r:r+1,...].to(config.device)
                measurements = noisy_dataset["measurements"][s:s+1,r:r+1,...].to(config.device)
                contamination_factor = noisy_dataset["contamination_factor"][s:s+1,r:r+1,...].to(config.device)
                attn_factors = noisy_dataset["attn_factors"][s:s+1,r:r+1,...].to(config.device) * detector_efficiency
                norm = get_normalisation(osem, measurements, contamination_factor)
                x_pred = torch.clamp(model(osem, measurements, projector, attn_factors, norm, contamination_factor).detach(),0)

                psnr_r.append(peak_signal_noise_ratio(reference.squeeze().cpu().numpy(), x_pred.squeeze().cpu().numpy(), data_range=reference.cpu().numpy().max()))
                ssim_r.append(ssim(reference.squeeze().cpu().numpy(), x_pred.squeeze().cpu().numpy(), data_range=reference.cpu().numpy().max()))

                y_pred = LPDForwardFunction2D.apply(x_pred, projector, attn_factors) + contamination_factor[:, None, None]
                pnll_r.append(pnll(y_pred, measurements).item())
                kl = (measurements*torch.log(measurements/y_pred+1e-9)+ (y_pred-measurements)).sum()
                if kl.isnan():
                    print("KL is nan")
                    exit()
                kldiv_r.append(kl)

                if tumour:
                    if background.sum() != 0:
                        background_idx = np.nonzero(background.cpu().numpy())
                        b_bar = x_pred.cpu()[background_idx]
                        b_t = reference.cpu().numpy()[background_idx]
                        crc_t = []
                        for i in range(3):
                            if tumour_rois[:,[i],:,:].sum() != 0:
                                tumour_roi_idx = np.nonzero(tumour_rois[:,[i],:,:].cpu().numpy())
                                a_bar = x_pred.cpu().numpy()[tumour_roi_idx]
                                a_t = reference.cpu().numpy()[tumour_roi_idx]
                                crc_t.append((a_bar.mean()/b_bar.numpy().mean() - 1) / (a_t.mean()/b_t.mean() - 1))
                        b_bar_r.append(b_bar)
                        crc_r_t.append(torch.asarray(crc_t).mean())
            if tumour:
                crc_s_r_t.append(torch.stack(crc_r_t).cpu())
                std_s.append(torch.std(torch.stack(b_bar_r), dim=0).mean())
            ssim_s_r.append(torch.asarray(ssim_r))
            psnr_s_r.append(torch.asarray(psnr_r))
            pnll_s_r.append(torch.asarray(pnll_r))
            kldiv_s_r.append(torch.asarray(kldiv_r))

            if s in save_idxs:
                x_preds_save.append(x_pred.squeeze().cpu())
                x_ref_save.append(reference.squeeze().cpu())
        if tumour:
            torch.save({"images": torch.stack(x_preds_save).cpu(),
                        "ref": torch.stack(x_ref_save).cpu(),
                        "crc": torch.stack(crc_s_r_t).cpu(), 
                        "ssim": torch.stack(ssim_s_r).cpu(), 
                        "psnr": torch.stack(psnr_s_r).cpu(), 
                        "pnll": torch.stack(pnll_s_r).cpu(),
                        "kldiv": torch.stack(kldiv_s_r).cpu(),
                        "std": torch.stack(std_s).cpu()}, model_path + f"tumour_results_{noise_level}.pt")
        else:
            torch.save({"images": torch.stack(x_preds_save).cpu(),
                        "ref": torch.stack(x_ref_save).cpu(),
                        "ssim": torch.stack(ssim_s_r).cpu(), 
                        "pnll": torch.stack(pnll_s_r).cpu(),
                        "kldiv": torch.stack(kldiv_s_r).cpu(),
                        "psnr": torch.stack(psnr_s_r).cpu()}, model_path + f"non_tumour_results_{noise_level}.pt")
    