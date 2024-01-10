import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
    # Create empty placeholder for A
    A = np.empty((I.shape[0] * I.shape[1], 6))
    # Fill in A value with x^2 xy y^2 x y 1 where x and y are the calibrated pixel value
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            x =((i+0.5)/I.shape[0]) - 0.5
            y =((j+0.5)/I.shape[1]) - 0.5
            A[i * I.shape[1] + j, :] = np.array([x**2, x*y, y**2, x, y, 1])
    
    # Create empty placeholder for b
    b = np.empty((I.shape[0] * I.shape[1], 1))
    # Fill in b value with intensity from I
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            b[i * I.shape[1] + j, :] = I[i, j]

    # Solve for linear least squares and extract the coefficients
    coeff = lstsq(A, b)[0]
    alpha = coeff[0]
    beta = coeff[1]
    gamma = coeff[2]
    delta = coeff[3]
    epsilon = coeff[4]
    xi = coeff[5]

    # Find the saddle point
    tmp_1 = -np.linalg.inv(np.array([[2 * alpha, beta], [beta, 2 * gamma]]).reshape(2, 2))
    tmp_2 = np.array([delta, epsilon])
    pt = np.matmul(tmp_1, tmp_2)

    # Adjust the point to be relative to the upper left corner of the patch
    x, y = pt[0] * I.shape[0]  + I.shape[0]//2, pt[1] * I.shape[1] + I.shape[1]// 2
    pt[0], pt[1] = y, x
 
    #------------------
    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt