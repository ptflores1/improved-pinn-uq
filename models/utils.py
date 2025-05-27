import os
import torch
import numpy as np
from scipy.interpolate import interpn

from models.error_bounds import get_bound_until, get_bounds_for_params


# Needs to be a class to be picklable
class StdGetter:
    def __init__(self, std, t_0, number_of_stds=None) -> None:
        self.std = std
        self.t_0 = t_0
        self.number_of_stds = number_of_stds
    
    def __call__(self, coordinates: [torch.Tensor]):
        t = coordinates[0]
        stds = torch.ones(t.shape) * self.std
        stds[t.ravel() == self.t_0]  = torch.finfo(stds.dtype).eps
        if self.number_of_stds is not None:
            return [stds] * self.number_of_stds
        return stds
    
class StdGetterEB:
    def __init__(self, z_0, z_f, int_eP, eP, f, bounded_params=None, number_of_stds=None, device=None, experiment_name=None) -> None:
        self.z_0 = z_0
        self.number_of_stds = number_of_stds
        self.bounded_params = bounded_params
        self.device = device


        if bounded_params is not None:
            if experiment_name is not None and os.path.exists(f"checkpoints/ebparams_t_{experiment_name}.npy"):
                print("Loading bounds from:", f"checkpoints/ebparams_t_{experiment_name}.npy", "and", f"checkpoints/ebparams_bounds_{experiment_name}.npy")
                self.t_bounds = np.load(f"checkpoints/ebparams_t_{experiment_name}.npy")
                self.bounds = np.load(f"checkpoints/ebparams_bounds_{experiment_name}.npy")
                self.interpolation_domain = (self.t_bounds.ravel(), ) + tuple(p.ravel() for p in self.bounded_params.T)
            else:
                params_grid = np.array(np.meshgrid(*bounded_params.T))
                params_grid = np.moveaxis(params_grid, 0, -1).reshape(-1, bounded_params.shape[1])
                self.t_bounds, self.bounds = get_bounds_for_params(params_grid, f, z_0, z_f, 100, int_eP, eP)
                self.t_bounds, self.bounds = self.t_bounds.cpu().numpy(), self.bounds.cpu().numpy()
                self.bounds = np.moveaxis(self.bounds, 0, 1).reshape(-1, *[bounded_params.shape[0]]*bounded_params.shape[1])

                duplicates = [i for i, t_b in enumerate(self.t_bounds[:-1]) if t_b == self.t_bounds[i+1]]
                self.t_bounds = np.delete(self.t_bounds, duplicates)
                self.bounds = np.delete(self.bounds, duplicates, axis=0)
                self.interpolation_domain = (self.t_bounds.ravel(), ) + tuple(p.ravel() for p in self.bounded_params.T)
                if experiment_name is not None:
                    print("Saving bounds in:", f"checkpoints/ebparams_t_{experiment_name}.npy", "and", f"checkpoints/ebparams_bounds_{experiment_name}.npy")
                    np.save(f"checkpoints/ebparams_t_{experiment_name}.npy", self.t_bounds)
                    np.save(f"checkpoints/ebparams_bounds_{experiment_name}.npy", self.bounds)
        else:
            if experiment_name is not None and os.path.exists(f"checkpoints/eb_t_{experiment_name}.npy"):
                print("Loading bounds from:", f"checkpoints/eb_t_{experiment_name}.npy", "and", f"checkpoints/eb_bounds_{experiment_name}.npy")
                self.t_bounds = np.load(f"checkpoints/eb_t_{experiment_name}.npy")
                self.bounds = np.load(f"checkpoints/eb_bounds_{experiment_name}.npy")
                self.interpolation_domain = (self.t_bounds.ravel(),)
            else:
                self.t_bounds, self.bounds = get_bound_until(f, z_0, z_f, 200, int_eP, eP)
                self.t_bounds, self.bounds = self.t_bounds.cpu().numpy(), self.bounds.cpu().numpy()
                duplicates = [i for i, t_b in enumerate(self.t_bounds[:-1]) if t_b == self.t_bounds[i+1]]
                self.t_bounds = np.delete(self.t_bounds, duplicates)
                self.bounds = np.delete(self.bounds, duplicates)
                self.interpolation_domain = (self.t_bounds.ravel(),)
                if experiment_name is not None:
                    print("Saving bounds in:", f"checkpoints/eb_t_{experiment_name}.npy", "and", f"checkpoints/eb_bounds_{experiment_name}.npy")
                    np.save(f"checkpoints/eb_t_{experiment_name}.npy", self.t_bounds)
                    np.save(f"checkpoints/eb_bounds_{experiment_name}.npy", self.bounds)

    def __call__(self, coordinates):
        points = torch.hstack([torch.tensor(c, device=self.device) for c in coordinates])
        if coordinates[0].shape[1] != 1:
            points = torch.hstack([c.reshape(-1, 1) for c in coordinates])
        eps = torch.sqrt(torch.tensor(torch.finfo().eps)).item()
        interp_cov_diagonal = torch.from_numpy(interpn(self.interpolation_domain, self.bounds, points.detach().cpu().numpy())).reshape(-1, 1).to(self.device)
        interp_cov_diagonal = torch.clamp(interp_cov_diagonal, min=eps)
        interp_cov_diagonal[coordinates[0].ravel() == self.z_0] = eps
        #interp_cov_diagonal += 0.005
        if coordinates[0].shape[1] != 1:
            interp_cov_diagonal = interp_cov_diagonal.reshape(coordinates[0].shape)
        if self.number_of_stds is not None:
            return [interp_cov_diagonal] * self.number_of_stds
        return interp_cov_diagonal
    