import torch

class Normalisation:
    def __init__(self, type_norm):
        if type_norm == "none":
            self.norm = self.none
        elif type_norm == "data_mean":
            self.norm = self.data_mean
        elif type_norm == "data_quantile":
            self.norm = self.data_quantile
        elif type_norm == "data_max":
            self.norm = self.data_quantile
        elif type_norm == "osem_mean":
            self.norm = self.osem_mean
        elif type_norm == "osem_quantile":
            self.norm = self.osem_quantile
        elif type_norm == "osem_max":
            self.norm = self.osem_max
        else:
            raise ValueError("normalisation type not recognised")

    def __call__(self, osem, measurements, contamination_factor):
        return self.norm(osem, measurements, contamination_factor)
    
    def none(self, osem, measurements, contamination_factor):
        norm = torch.ones_like(contamination_factor[:,0])
        return norm
    
    def data_mean(self, osem, measurements, contamination_factor):
        norm = (measurements - contamination_factor[...,None]).mean(dim=[1,2])
        print(norm.shape)
        return norm
    
    def data_quantile(self, osem, measurements, contamination_factor):
        norm = torch.quantile((measurements - contamination_factor[...,None]).view(osem.shape[0],-1), q = 0.99, dim=1)
        return norm
    
    def data_max(self, osem, measurements, contamination_factor):
        norm = (measurements - contamination_factor[...,None]).max(dim=[1,2])
        return norm
    
    def osem_mean(self, osem, measurements, contamination_factor):
        norm = osem.mean(dim=[1,2,3])
        return norm
    
    def osem_quantile(self, osem, measurements, contamination_factor):
        norm = torch.quantile(osem.view(osem.shape[0],-1), q = 0.99, dim=1)
        return norm
    
    def osem_max(self, osem, measurements, contamination_factor):
        norm = osem.max(dim=[1,2,3])
        return norm
