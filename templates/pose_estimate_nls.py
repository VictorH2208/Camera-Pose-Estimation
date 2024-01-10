import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from support.dcm_from_rpy import dcm_from_rpy
from support.rpy_from_dcm import rpy_from_dcm

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

def pose_estimate_nls(K, Twc_guess, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    K          - 3x3 camera intrinsic calibration matrix.
    Twc_guess  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts       - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts       - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array (float64), pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    iter = 1

    #--- FILL ME IN ---

    # 1. Convert initial guess to parameter vector (6 x 1).
    Twc = Twc_guess
    params = epose_from_hpose(Twc)

    # 2. Main loop - continue until convergence or maxIters.
    while True:
        # 3. Save previous best pose estimate.
        params_prev = params
        Twc = hpose_from_epose(params_prev)

        R = Twc[:3, :3]
        t = Twc[:3, 3]

        # 4. Project each landmark into image, given current pose estimate.
        for i in np.arange(tp):
            # Compute residuals.
            residual = K @ R.T @ (Wpts[:, i]-t) 
            residual = (residual/residual[-1])[0:2]
            dY[2*i:2*i+2, :] = residual.reshape(2, 1) - Ipts[:,i].reshape(2, 1)

            # Compute Jacobian.
            Jacobian = find_jacobian(K, Twc, Wpts[:,i].reshape(3,1))
            J[2*i:2*i+2, :] = Jacobian

        # 5. Solve system of normal equations for this iteration.
        delta = inv(J.T @ J) @ J.T @ dY
        params = params_prev - delta

        # 6. Check - converged?
        if norm(delta) < 1e-12:
            print("Covergence required %d iters." % iter)
            break
        elif iter == maxIters:
            print("Failed to converge after %d iters." % iter)
            break
        
        iter += 1

    # 7. Compute and return homogeneous pose matrix Twc.
    Twc = hpose_from_epose(params)

    #------------------

    correct = isinstance(Twc, np.ndarray) and \
        Twc.dtype == np.float64 and \
        Twc.shape == (4, 4) and Twc[3, 3] == 1.0000

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Twc
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    Twcg  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts  - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts  - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array, homogenous pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---
    Twc = Twcg
    
    for i in range(maxIters):
            R = Twc[:3, :3]
            t = Twc[:3, 3]
            for j in range(tp):
                    # e(x) = K * R^T * (Wpt - t) - Ipt
                    error = K @ R.T @ (Wpts[:,j]-t) 
                    error /= error[-1] 

                    e = error[:-1].reshape(2,1)
                    I = Ipts[:,j].reshape(2,1)
                    
                    dY[j:j+2,:] = e - I
                    
                    # compute jacobian matrix
                    Jacobian = find_jacobian(K, Twc, Wpts[:,j].reshape(3,1))
                    J[j:j+2,:] = Jacobian
            # solve normal equations
            dx = -inv(J.T @ J) @ J.T @ dY
            # converged? 
            if norm (dx) < 1e-4:
                    break
            # update parameters 
            x = epose_from_hpose(Twc)
            x+=dx
            Twc = hpose_from_epose(x)

    #------------------
    
    return Twc