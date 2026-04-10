import numpy as np
from numba import njit, prange

@njit
def rhohl(sbeg, send, pt):
    l = send - sbeg
    l_norm = np.linalg.norm(l)
    l_hat = l / l_norm
    R = pt - send
    a = np.cross(l_hat, R)
    rho_hat = np.cross(a, l_hat) / np.linalg.norm(a)
    rho = np.dot(R, rho_hat)
    h = np.dot(R, l_hat)
    return rho_hat, rho, h, l_hat, l_norm + h

@njit
def unit_magnetic_field(pt, sbeg, send):
    """Calculate the magnetic field produced by a unit current along the segment sbeg-send at point in space `pt`.
    
    ..math::
    \mathbf{B} = \frac{\mu_0}{4\pi} \sum_n \frac{\hat{\mathbf{I}} \times \hat{\boldsymbol{\rho}}_n}{\rho_n} \left[\frac{h_n}{\sqrt{h_n^2 + \rho_n^2}} - \frac{l_n}{\sqrt{l_n^2 + \rho_n^2}}\right] d\ell
    
    where:
    * :math:`\hat{\mathbf{I}}` is the unit current vector (dimensionless!)
    * :math:`\hat{\boldsymbol{\rho}}_n` is the unit vector perpendicular to the segment
    * :math:`h_n, l_n` are the distances to the segment endpoints along its axis
    * :math:`\rho_n` is the perpendicular distance to the segment
    * :math:`d\ell` is the differential length of the segment
    
    `pt` is the observation point in space.
    `sbeg` and `send` are the start and end points of the segment.
    
    Returns: 
        unit_B: (numpy.ndarray) the unit magnetic field vector in uT / A 
                if units of `pt` and `sbeg`/`send` are in meters.
                Units: 
                    [output] = (uT * m / A) / [distance], 
                    where [distance] is the unit of measure of `pt` and `sbeg`/`send`, e.g. meters.
    """
    rho_hat, rho, h, l_hat, l_norm = rhohl(sbeg, send, pt)
    factor = (h / np.sqrt(h**2 + rho**2) - l_norm / np.sqrt(l_norm**2 + rho**2)) / rho
    unit_B = 1e-1 * np.cross(l_hat, rho_hat) * factor  
    #            |
    #         1e-1 uT * m / A = μ0 / 4π in the prior to 2019 defintion of μ0
    return unit_B  # units of (uT * m / A) / [distance]

@njit(parallel=True)
def get_b_njit(current_pts, pts, current):
    """Calculate the magnetic field produced by a current going through `current_pts` at
    npts points in space `pts`.
    
    `current_pts` should be of shape (N, 3) where N is the number of points in the current path, and
    `current` should be of shape (N-1, T) where T is the number of time points. `current` direction
    is defined by the direction from `current_pts[i]` to `current_pts[i+1]`.
    
    Returns:
        B: (numpy.ndarray) the magnetic field vector in uT if units of `pts` are in meters and `current` is in A.
            Dimension is (3, npts, T). 
            Units: 
                [output] = (uT * m / A) * [current] / [distance], 
                where [distance] is the unit of measure of both `pts` and `current_pts`, e.g. meters.
    """
    npts = len(pts)
    nsegs = len(current_pts) - 1
    nt = current.shape[1]
    B = np.zeros((3, npts, nt))
    
    for i in prange(npts):
        pt = pts[i]
        for j in prange(nsegs):
            p1 = current_pts[j]
            p2 = current_pts[j+1]
            unit_B = unit_magnetic_field(pt, p1, p2)
            B[:, i, :] += np.outer(unit_B, current[j])
    return B  # units of (uT * m / A) * [current] / [distance]

@njit(parallel=True, fastmath=True)
def get_odmr_shifts(B_fields, nv_axes, gamma):
    """Calculate ODMR frequency shifts for B-field vectors and NV orientations.
    
    Efficiently handles different input shapes:
    - Single B field vector: shape (3,)
    - Multiple B field vectors: shape (n_points, 3)
    
    Args:
        B_fields: Magnetic field vector(s)
        nv_axes: NV axes, shape (n_axes, 3)
        gamma: Gyromagnetic ratio in Hz/T
        
    Returns:
        shifts: Frequency shifts with shape:
               - (n_axes,) for single B field input
               - (n_points, n_axes) for multiple B fields
    """
    # Check input dimensionality
    is_single = B_fields.ndim == 1
    n_axes = len(nv_axes)
    
    # Handle single B field vector (3,)
    if is_single:
        shifts = np.empty(n_axes, dtype=np.float64)
        
        # Unroll the loop for better performance
        for a in prange(n_axes):
            # Use explicit dot product computation
            B_proj = (B_fields[0] * nv_axes[a, 0] + 
                     B_fields[1] * nv_axes[a, 1] + 
                     B_fields[2] * nv_axes[a, 2])
            shifts[a] = 2.0 * gamma * B_proj
            
        return shifts
        
    # Handle multiple B field vectors (n_points, 3)
    else:
        n_points = len(B_fields)
        shifts = np.empty((n_points, n_axes), dtype=np.float64)
        
        # Pre-compute constants
        factor = 2.0 * gamma
        
        # Parallelize over spatial points
        for p in prange(n_points):
            # Cache B field components for better memory access
            b_x = B_fields[p, 0]
            b_y = B_fields[p, 1]
            b_z = B_fields[p, 2]
            
            for a in prange(n_axes):
                # Manually unrolled dot product for better performance
                B_proj = (b_x * nv_axes[a, 0] + 
                         b_y * nv_axes[a, 1] + 
                         b_z * nv_axes[a, 2])
                shifts[p, a] = factor * B_proj
                
        return shifts
    
@njit(parallel=True, fastmath=True)
def get_multipole_expansion(currents, pts, order=0, r0=None):
    """Compute n-th order of multipole expansion of the current. 
    Current is a list of values of the current in units of current (A, mA) and `pts` are 
    points between which the current is flowing.
    
    currents: (array-like of length (N-1,)) list of current values in units of current (A, mA)
    pts: (array-like of length (N, 3)) list of points between which the current is flowing
    order: (int) order of the multipole expansion
    r0: (array-like of length (3)) reference point for the multipole expansion, default: (0, 0, 0)
    
    Returns:
        (a vector of length (n_order * 3)) n-th order of multipole expansion of the current
    """
    npts = len(pts)
    nsegs = len(currents)
    
    assert npts - 1 == nsegs, "Number of points must be one more than number of segments"
    
    # Pre-allocate arrays
    current_vecs = np.zeros((3, nsegs))
    result = np.zeros(3)
    
    # Compute current vectors
    for i in prange(nsegs):
        current_vecs[0, i] = currents[i] * (pts[i+1, 0] - pts[i, 0])
        current_vecs[1, i] = currents[i] * (pts[i+1, 1] - pts[i, 1])
        current_vecs[2, i] = currents[i] * (pts[i+1, 2] - pts[i, 2])
    
    if order == 0:
        """Compute monopole expansion, i.e. the vector sum of all the currents."""
        for i in prange(3):
            result[i] = np.sum(current_vecs[i, :])
        return result
    
    elif order == 1:
        """Compute dipole expansion, i.e. the vector sum of 
        
        \sum_{i=1}^{N-1} \vec{I}_i × (pm_i - r0),
        
        where \vec{I}_i is the computed directed current in `current_vecs`,
        and pm_i is the midpoint between points i and i+1.
        """
        midpoints = np.zeros((nsegs, 3))
        for i in prange(nsegs):
            midpoints[i, 0] = 0.5 * (pts[i, 0] + pts[i+1, 0])
            midpoints[i, 1] = 0.5 * (pts[i, 1] + pts[i+1, 1])
            midpoints[i, 2] = 0.5 * (pts[i, 2] + pts[i+1, 2])
        
        # Compute dipole expansion
        for i in prange(nsegs):
            # Compute relative position
            rx = midpoints[i, 0] - r0[0]
            ry = midpoints[i, 1] - r0[1]
            rz = midpoints[i, 2] - r0[2]
            
            # Compute cross product manually for better performance
            result[0] += current_vecs[1, i] * rz - current_vecs[2, i] * ry
            result[1] += current_vecs[2, i] * rx - current_vecs[0, i] * rz
            result[2] += current_vecs[0, i] * ry - current_vecs[1, i] * rx
            
        return result
    
    else:
        raise NotImplementedError(f"Order {order} is not implemented yet")