import numpy as np
import jpegio as jio
from tqdm import tqdm
import os
from scipy import fftpack
from numpy.lib.stride_tricks import as_strided
from collections import defaultdict 

def block_view(A, block= (8,8)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]// block[0], A.shape[1]// block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return as_strided(A, shape= shape, strides= strides)

def segmented_stride(M, fun, blk_size=(8,8), overlap=(0,0)):
    # This is some complex function of blk_size and M.shape
    B = block_view(M, block=blk_size)
    B[:,:,:,:] = fun(B)
    return M

def decompress_structure(S):
    # Decompress DCT coefficients C using quantization table Q
    H = S.coef_arrays[0].shape[0]
    W = S.coef_arrays[0].shape[1]
    n = len(S.coef_arrays)
    assert H % 8 == 0, 'Wrong image size'
    assert W % 8 == 0, 'Wrong image size'
    I = np.zeros((H,W,n),dtype=np.float64) # Returns Y, Cb and Cr
    for i in range(n):
        Q = S.quant_tables[S.comp_info[i].quant_tbl_no]
        # this multiplication is done on integers
        fun = lambda x : np.multiply(x,Q)
        C = np.float64(segmented_stride(S.coef_arrays[i], fun)) 
        fun = lambda x: fftpack.idct(fftpack.idct(x, norm='ortho',axis=2), norm='ortho',axis=3) + 128
        I[:,:,i] = segmented_stride(C, fun)
    return I