
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
from src import BrainWebOSEM
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
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
                reference = reference.to("cuda")

                scale_factor = batch[1]
                scale_factor = scale_factor.to("cuda")

                osem = batch[2]
                osem = osem.to("cuda")

                norm = batch[3]
                norm = norm.to("cuda")

                osem = osem/norm[...,None,None,None]

                
                psnr_pred_list.append(peak_signal_noise_ratio(reference.cpu().numpy()[0,0,:,:], osem.cpu().numpy()[0,0,:,:], data_range=reference.cpu().numpy().max()))
                ssim_pred_list.append(ssim(reference.cpu().numpy()[0,0,:,:], osem.cpu().numpy()[0,0,:,:], data_range=reference.cpu().numpy().max()))

                if "tumour" in part:
                    tumour_rois = batch[-1]
                    tumour_rois = tumour_rois[0,...].cpu().numpy()
                    background = batch[-2]
                    background = background.squeeze().cpu().numpy()
                    if background.sum() != 0:
                        background_idx = np.nonzero(background)
                        b_bar = osem.squeeze().cpu().numpy()[background_idx]
                        b_t = reference.squeeze().cpu().numpy()[background_idx]
                        for tumour_roi in tumour_rois:
                            if tumour_roi.sum() != 0:
                                tumour_roi_idx = np.nonzero(tumour_roi)
                                a_bar = osem.squeeze().cpu().numpy()[tumour_roi_idx]
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
    torch.save(qm_dicts, "OSEM_EVAL.pt")






