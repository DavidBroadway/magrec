import torch
import torch.optim as optim

from magrec.prop.Propagator import MagneticDipolePropagator, AxisProjectionPropagator
from magrec.misc.data import Scaler

from scipy.spatial import cKDTree

import numpy as np

class Trainer:
    """
    Fits dipole moments to reproduce observed B_NV field. Builds FFM once, then Adam to minimize MSE.
    Optional physics regularization (run(..., lambda_tv=, lambda_ex=, lambda_dm=)):
    - TV: total variation on m over neighbor edges → piecewise constant, sharp flips at long boundaries.
    - Exchange: ferromagnetic-style (1 - m_i·m_j), capped so domain walls are allowed.
    - DM: Dzyaloshinskii-Moriya D·(m_i × m_j) for chirality. Spin-glass = random J_ij (not implemented; add _reg_sg if needed).
    Neighbors from grid_shape=(nx,ny) if full grid, else from geometry (4-neighbors in xy).
    """
    
    dtype = torch.float32  # change to float64 if precision issues arise
    
    def __init__(self, region, r_sensor, r_source, B_NV, theta=54.7, phi=-45.0, grid_shape=None):
        # Select points in region, convert everything to proper dtype
        if region is not None:
            self.region = region
            r_sensor_selected, self.idx = region.select(r_sensor)
            r_source_selected, _ = region.select(r_source)
        else:
            r_sensor_selected = r_sensor
            r_source_selected = r_source
        
        self.r_sensor = self._to_tensor(r_sensor_selected)
        self.r_source = self._to_tensor(r_source_selected)
        self.n_dipoles = r_source_selected.shape[0]
        self.grid_shape = grid_shape  # (nx, ny) optional; else edges from geometry
        
        self.B_NV = self._to_tensor(B_NV)
        
        # Target field: flatten B_NV grid, select by same indices, scale to unit variance
        B_flat = self.B_NV.flatten()[self.idx]
        self.scaler = Scaler(B_flat)
        self.target = self._to_tensor(self.scaler.scale(B_flat))
        
        # Check if the size of the matrix is too large
        if MagneticDipolePropagator.get_expected_ffm_size(source=self.r_source, sensor=self.r_sensor, out_units="MB") > 100:
            raise RuntimeError(f"Too large matrix expected for the number of source locations {self.r_source.shape[0]} and" + \
                              f"number of sensor locations {self.r_sensor.shape[0]}")
        
        # Build forward-field matrix (expensive, do once). Shape: (n_sensor, 3, n_source, 3)
        # FFM[i,j,k,l] = B_j at sensor i from unit dipole_l at source k
        self.prop = MagneticDipolePropagator(r_source=self.r_source, r_sensor=self.r_sensor, backend="torch")
        
        # NV projection axis
        self.projector = AxisProjectionPropagator(theta=theta, phi=phi)
        self.n_NV = self.projector.n.to(self.dtype)
        
        # Neighbor edges for physics regularization (TV, exchange, DM). Built once from grid or geometry.
        self._edges = self._build_edges()
        
        # Parameters to optimize: (n_dipoles, 3) moments, small random init
        self.dipole_moments = torch.randn(self.n_dipoles, 3, dtype=self.dtype) * 0.01
        self.dipole_moments = self.dipole_moments.clone().detach().requires_grad_(True)
        
        self.loss_history = []
        # Per-term histories for plotting and tuning lambdas (same length as loss_history after run)
        self.loss_data_history = []
        self.loss_tv_history = []
        self.loss_ex_history = []
        self.loss_dm_history = []
        self.loss_sg_history = []
        self._total_iters = 0
    
    def _build_edges(self):
        """Pairs (i, j) of neighbor indices for regularization. From grid_shape if given, else from geometry (4-neighbors on xy)."""
        n = self.n_dipoles
        r = self.r_source.detach().cpu().numpy()
        xy = r[:, :2]
        if self.grid_shape is not None:
            nx, ny = self.grid_shape
            if nx * ny != n:
                return []
            edges = []
            for iy in range(ny):
                for ix in range(nx):
                    k = iy * nx + ix
                    if ix + 1 < nx:
                        edges.append((k, k + 1))
                    if iy + 1 < ny:
                        edges.append((k, k + nx))
            return edges
        ux = np.unique(np.sort(xy[:, 0]))
        uy = np.unique(np.sort(xy[:, 1]))
        dx = float(np.median(np.diff(ux))) if len(ux) > 1 else 1.0
        dy = float(np.median(np.diff(uy))) if len(uy) > 1 else 1.0
        if dx <= 0 or not np.isfinite(dx):
            dx = dy if np.isfinite(dy) and dy > 0 else 1.0
        if dy <= 0 or not np.isfinite(dy):
            dy = dx
        d_lo, d_hi = 0.4 * min(dx, dy), 1.6 * max(dx, dy)
        tree = cKDTree(xy)
        edges = set()
        for i in range(n):
            for j in tree.query_ball_point(xy[i], r=d_hi):
                if j <= i:
                    continue
                d = np.linalg.norm(xy[i] - xy[j])
                if d_lo <= d <= d_hi and (np.abs(d - dx) < 0.4 * dx or np.abs(d - dy) < 0.4 * dy):
                    edges.add((i, j))
        return list(edges)
    
    def _reg_tv(self, m):
        """Total variation: sum over edges of |m_i - m_j|. Promotes piecewise constant m; allows sharp flips at few boundaries."""
        if not self._edges:
            return m.new_zeros(())
        i, j = zip(*self._edges)
        i, j = torch.tensor(i, dtype=torch.long, device=m.device), torch.tensor(j, dtype=torch.long, device=m.device)
        return (m[i] - m[j]).norm(dim=1).mean()
    
    def _reg_exchange(self, m, cap=2.0):
        """Ferromagnetic-style: sum (1 - m_i·m_j) over edges, capped so full flips are not infinitely penalized."""
        if not self._edges:
            return m.new_zeros(())
        i, j = zip(*self._edges)
        i, j = torch.tensor(i, dtype=torch.long, device=m.device), torch.tensor(j, dtype=torch.long, device=m.device)
        dot = (m[i] * m[j]).sum(dim=1)
        return (cap - torch.clamp(dot, -1.0, 1.0)).mean()
    
    def _reg_dm(self, m, bond_vectors=None):
        """Dzyaloshinskii-Moriya: sum D_ij·(m_i × m_j). D_ij from bond direction (in-plane) if bond_vectors not given."""
        if not self._edges:
            return m.new_zeros(())
        i, j = zip(*self._edges)
        i, j = torch.tensor(i, dtype=torch.long, device=m.device), torch.tensor(j, dtype=torch.long, device=m.device)
        cross = torch.linalg.cross(m[i], m[j], dim=1)
        if bond_vectors is not None:
            D = bond_vectors.to(m.device)
        else:
            r = self.r_source
            D = (r[j] - r[i])[:, :2]
            D = torch.nn.functional.pad(D, (0, 1))  # z=0 so D in-plane
        return (D * cross).sum(dim=1).abs().mean()
    
    def _reg_sg(self, m, J=None):
        """Spin-glass: sum_edges J_ij (m_i·m_j). J (num_edges,) random; fixed at first call if None."""
        if not self._edges:
            return m.new_zeros(())
        if not hasattr(self, '_J_sg') or self._J_sg is None:
            self._J_sg = (torch.randn(len(self._edges), device=m.device, dtype=m.dtype) * 0.1) if J is None else J.to(m.device)
        i, j = zip(*self._edges)
        i, j = torch.tensor(i, dtype=torch.long, device=m.device), torch.tensor(j, dtype=torch.long, device=m.device)
        dot = (m[i] * m[j]).sum(dim=1)
        return (self._J_sg.to(m.device) * dot).mean()
    
    def _to_tensor(self, x):
        """Convert numpy/torch array to tensor with proper dtype."""
        if isinstance(x, torch.Tensor):
            return x.to(self.dtype)
        return torch.tensor(x, dtype=self.dtype)
    
    def forward(self):
        """Compute predicted B_NV from current dipole moments."""
        # einsum: contract source index and moment component with FFM
        B_pred = self.prop(self.dipole_moments)  # (n_sensor, 3)
        B_NV_pred = torch.einsum('ij,j->i', B_pred, self.n_NV)              # (n_sensor,)
        return B_NV_pred
    
    def run(self, n_iters=1000, lr=1e-6, print_every=100, lambda_tv=0.0, lambda_ex=0.0, lambda_dm=0.0, lambda_sg=0.0):
        """Run optimization. Regularization: lambda_tv (TV), lambda_ex (exchange), lambda_dm (DM), lambda_sg (spin-glass)."""
        opt = optim.Adam([self.dipole_moments], lr=lr)
        m = self.dipole_moments
        
        for i in range(n_iters):
        
            opt.zero_grad()
            pred = self.forward()
            loss_data = ((pred - self.target) ** 2).mean()
        
            r_tv = self._reg_tv(m) if lambda_tv != 0 else m.new_zeros(())
            r_ex = self._reg_exchange(m) if lambda_ex != 0 else m.new_zeros(())
            r_dm = self._reg_dm(m) if lambda_dm != 0 else m.new_zeros(())
            r_sg = self._reg_sg(m) if lambda_sg != 0 else m.new_zeros(())
        
            self.loss_data_history.append(loss_data.item())
            self.loss_tv_history.append(r_tv.item() if r_tv.numel() else 0.0)
            self.loss_ex_history.append(r_ex.item() if r_ex.numel() else 0.0)
            self.loss_dm_history.append(r_dm.item() if r_dm.numel() else 0.0)
            self.loss_sg_history.append(r_sg.item() if r_sg.numel() else 0.0)
        
            loss = loss_data + lambda_tv * r_tv + lambda_ex * r_ex + lambda_dm * r_dm + lambda_sg * r_sg
            loss.backward()
            opt.step()
            self.loss_history.append(loss.item())
            if print_every and i % print_every == 0:
                print(f"iter {self._total_iters + i:5d}  loss {loss.item():.4e}")
        self._total_iters += n_iters
        return self
    
    def predict_full(self):
        """Get predicted B_NV in original (unscaled) units."""
        with torch.no_grad():
            pred_scaled = self.forward()
        return self.scaler.unscale(pred_scaled)
    
    def plot_losses(self, fig=None, ax=None, logy=True):
        """Plot each loss term over iterations to compare contributions and tune lambdas."""
        import matplotlib.pyplot as plt
        n = len(self.loss_data_history)
        if n == 0:
            return
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4)) if fig is None else (fig, fig.gca())
        iters = range(n)
        ax.plot(iters, self.loss_data_history, label='data (MSE)', color='C0')
        if any(self.loss_tv_history):
            ax.plot(iters, self.loss_tv_history, label='TV', color='C1')
        if any(self.loss_ex_history):
            ax.plot(iters, self.loss_ex_history, label='exchange', color='C2')
        if any(self.loss_dm_history):
            ax.plot(iters, self.loss_dm_history, label='DM', color='C3')
        if any(self.loss_sg_history):
            ax.plot(iters, self.loss_sg_history, label='spin-glass', color='C4')
        ax.plot(iters, self.loss_history, label='total', color='k', linestyle='--', alpha=0.7)
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        if logy:
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig, ax