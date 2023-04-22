import torch 

class BrainWebSupervisedTrain(torch.utils.data.Dataset):
    def __init__(self, noise_level, base_path="E:/projects/pyparallelproj/examples/data/", device="cpu", guided=False):
        assert noise_level in [5, 10, 50, "5", "10", "50"], "noise level has to be 5, 10, 50"
        self.base_path = base_path

        # dict_keys(['clean_measurements', 'mu', 'reference', 'mri]) 
        clean = torch.load(base_path+"clean/clean_train.pt", map_location=torch.device(device))
        # dict_keys(['osem', 'scale_factor', 'measurements', 'contamination_factor', 'attn_factors'])
        self.noisy = torch.load(base_path+"noisy/noisy_train_"+str(noise_level)+".pt", map_location=torch.device(device))
        self.reference = clean["reference"]
        self.guided = guided
        if self.guided:
            self.mri = clean["mri"]
    def __len__(self):
        return len(self.reference)

    def __getitem__(self, idx):
        reference = self.reference[idx,...].float()*self.noisy["scale_factor"][idx][None]

        osem = self.noisy["osem"][idx, ...].float()

        measurements = self.noisy["measurements"][idx, ...].float()

        contamination_factor = self.noisy["contamination_factor"][idx]

        attn_factors = self.noisy["attn_factors"][idx, ...].float()
        
        if self.guided:
            mri = self.mri[idx,...].float()
            return reference, mri, osem, measurements, contamination_factor, attn_factors
        # ref, osem, measurements, contamination_factor, attn_factors
        return reference, osem, measurements, contamination_factor, attn_factors