import nibabel as nib
import pyparallelproj.coincidences as coincidences
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
from tqdm import tqdm
import cupy as xp
import torch, os
import cupyx.scipy.ndimage as ndi
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__=="__main__":
    coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
                                    num_rings=1,
                                    sinogram_spatial_axis_order=coincidences.
                                    SinogramSpatialAxisOrder['RVP'],
                                    xp=xp)
    
    mu_projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                                (256,256,1), 
                                                (-127.5, -127.5, 0),
                                                (1,1,2))
    true_projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                                (256,256,1), 
                                                (-127.5, -127.5, 0),
                                                (1,1,2))
    
    res_model = resolution_models.GaussianImageBasedResolutionModel((256,256,1), 
                tuple(4.5 / (2.35 * x) for x in (1,1,2)), xp, ndi)
    
    true_projector.image_based_resolution_model = res_model
    
    subjects = [5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

    clean_data_pts = []
    mu_ref_pts = []
    image_ref_pts = []
    mri_ref_pts = []
    for subject_number in tqdm(subjects):
        for sim_number in range(3):
            nii_pet = nib.as_closest_canonical(nib.load(f'E:/projects/pyparallelproj/examples/data/brainweb_petmr/subject{subject_number:02}/sim_{sim_number}/true_pet.nii.gz'))
            nii_mu = nib.as_closest_canonical(nib.load(f'E:/projects/pyparallelproj/examples/data/brainweb_petmr/subject{subject_number:02}/mu.nii.gz'))
            nii_mri = nib.as_closest_canonical(nib.load(f'E:/projects/pyparallelproj/examples/data/brainweb_petmr/subject{subject_number:02}/t1.nii.gz'))
            # pet image resolution [1,1,2] mm
            image_gt = xp.array(nii_pet.get_fdata(), dtype=xp.float32)
            image_gt = (image_gt[:, :, ::2] + image_gt[:, :, 1::2])/2

            # pet image resolution [2,2,2] mm
            image_ref = (image_gt[::2, :, :] + image_gt[1::2, :, :])/2
            image_ref = (image_ref[:, ::2, :] + image_ref[:, 1::2, :])/2

            # mu image resolution [1,1,2] mm
            mu_gt = xp.array(nii_mu.get_fdata(), dtype=xp.float32)
            mu_gt = (mu_gt[:, :, ::2] + mu_gt[:, :, 1::2]) /2

            # mu image resolution [2,2,2] mm
            mu_ref = (mu_gt[::2, :, :] + mu_gt[1::2, :, :])/2
            mu_ref = (mu_ref[:, ::2, :] + mu_ref[:, 1::2, :])/2

            # mri image resolution [1,1,2] mm
            mri_gt = xp.array(nii_mri.get_fdata(), dtype=xp.float32)
            mri_gt = (mri_gt[:, :, ::2] + mri_gt[:, :, 1::2]) /2

            # mri image resolution [2,2,2] mm
            mri_ref = (mri_gt[::2, :, :] + mri_gt[1::2, :, :])/2
            mri_ref = (mri_ref[:, ::2, :] + mri_ref[:, 1::2, :])/2

            for slice_number in range(image_ref.shape[-1]):
                # ENSURE THERE ARE AT LEAST 2000 NON-ZERO PIXELS IN SLICE
                if len(xp.nonzero(image_ref[:, :, slice_number])[0]) > 2000:

                    attenuation_factors = xp.exp(-mu_projector.forward(mu_gt[:, :, [slice_number]]))
                    true_projector.multiplicative_corrections = attenuation_factors * 1./30
                    clean_data = true_projector.forward(image_gt[:, :, [slice_number]])

                    clean_data_pts.append(torch.from_dlpack(clean_data[None][None]).float().cuda())
                    mu_ref_pts.append(torch.from_dlpack(mu_ref[:, :, slice_number][None][None]).float().cuda())
                    image_ref_pts.append(torch.from_dlpack(image_ref[:, :, slice_number][None][None]).float().cuda())
                    mri_ref_pts.append(torch.from_dlpack(mri_ref[:, :, slice_number][None][None]).float().cuda())

    clean_data_pts = torch.cat(clean_data_pts)
    mu_ref_pts = torch.cat(mu_ref_pts)
    image_ref_pts = torch.cat(image_ref_pts)
    mri_ref_pts = torch.cat(mri_ref_pts)
    recon_dict = {'clean_measurements': clean_data_pts, 'mu': mu_ref_pts, 'reference': image_ref_pts, 'mri': mri_ref_pts}
    torch.save(recon_dict, "E:/projects/pet_score_model/src/brainweb_2d/clean/clean_train.pt")



                
