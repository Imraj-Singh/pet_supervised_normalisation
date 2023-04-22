
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
from src import get_lpd_model, BrainWebOSEM, LPDForwardFunction2D, LPDAdjointFunction2D
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pyparallelproj import coincidences, resolution_models, petprojectors
import cupy as xp
import cupyx.scipy.ndimage as ndi

if __name__ == "__main__":
    config = omegaconf.OmegaConf.load(r"E:\projects\pet_score_model\lpd_coordinators\LPD\.hydra\config.yaml")
    model = get_lpd_model(n_iter = config.benchmark.n_iter, op = LPDForwardFunction2D, op_adj = LPDAdjointFunction2D)
    model.load_state_dict(torch.load(os.path.join(r"E:\projects\pet_score_model\lpd_coordinators\LPD\Apr04_17-47-46_Fiona\model_min_val_loss.pt")))
    model.eval()
    model.to(config.device)

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

    
    paths = sorted(glob("E:/projects/pyparallelproj/examples/data/noisy/*test*.pt"))
    names = [name.rsplit("\\")[-1][:-3] for name in paths]
    qm_dicts = {}
    for idx in range(len(names)):
        part = names[idx].rsplit("_",2)[0]
        noise_level = names[idx].rsplit("_",2)[-1]
        dataset = BrainWebOSEM(part=part, noise_level=noise_level)

        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        psnr_pred_list = []
        ssim_pred_list = [] 
        if "tumour" in part:
            crc_list = []
            stdev_list = []
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if torch.count_nonzero(batch[0]) > 20*20:    

                reference = batch[0]
                reference = reference.to(config.device)

                scale_factor = batch[1]
                scale_factor = scale_factor.to(config.device)

                osem = batch[2]
                osem = osem.to(config.device)

                norm = batch[3]
                norm = norm.to(config.device)

                measurements = batch[4]
                measurements = measurements.to(config.device)

                
                contamination_factor = batch[5]
                contamination_factor = contamination_factor.to(config.device)

                attn_factors = batch[6]
                attn_factors = attn_factors.to(config.device)*detector_efficiency

                with torch.no_grad():
                    x_pred = model(osem, measurements, projector, attn_factors, norm, contamination_factor)
                
                psnr_pred_list.append(peak_signal_noise_ratio(reference.cpu().numpy()[0,0,:,:], x_pred.cpu().numpy()[0,0,:,:], data_range=reference.cpu().numpy().max()))
                ssim_pred_list.append(ssim(reference.cpu().numpy()[0,0,:,:], x_pred.cpu().numpy()[0,0,:,:], data_range=reference.cpu().numpy().max()))

                if "tumour" in part:
                    tumour_rois = batch[-1]
                    tumour_rois = tumour_rois[0,...].cpu().numpy()
                    background = batch[-2]
                    background = background.squeeze().cpu().numpy()
                    if background.sum() != 0:
                        background_idx = np.nonzero(background)
                        b_bar = x_pred.squeeze().cpu().numpy()[background_idx]
                        b_t = reference.squeeze().cpu().numpy()[background_idx]
                        for tumour_roi in tumour_rois:
                            if tumour_roi.sum() != 0:
                                tumour_roi_idx = np.nonzero(tumour_roi)
                                a_bar = x_pred.squeeze().cpu().numpy()[tumour_roi_idx]
                                a_t = reference.squeeze().cpu().numpy()[tumour_roi_idx]
                                crc_list.append((a_bar.mean()/b_bar.mean() - 1) / (a_t.mean()/b_t.mean() - 1))
                        stdev_list.append(b_bar.std())


        qm_dict = {"psnr_pred": psnr_pred_list,
                     "ssim_pred": ssim_pred_list}
        print("PART: ", part)
        print("NOISE LEVEL: ", noise_level)
        print("\t \ PSNR mean: ", np.mean(psnr_pred_list),", std: ", np.std(psnr_pred_list))
        print("\t \ SSIM mean: ", np.mean(ssim_pred_list),", std: ", np.std(ssim_pred_list))

        if "tumour" in part:
            qm_dict["crc"] = crc_list
            qm_dict["stdev"] = stdev_list
            print("\t \ CRC mean: ", np.mean(crc_list),", std: ", np.std(crc_list))
            print("\t \ STDEV mean: ", np.mean(stdev_list),", std: ", np.std(stdev_list))
    
        qm_dicts[part + "_" + str(noise_level)] = qm_dict
    torch.save(qm_dicts, "LPD_EVAL.pt")







