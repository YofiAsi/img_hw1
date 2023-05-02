import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse

def set_D(mat):
    mat.setdiag(-1, -1)
    mat.setdiag(4)
    mat.setdiag(-1, 1)
    return mat

def set_minos_I(mat, offset):
    mat.setdiag(-1, offset)
    mat.setdiag(-1, -offset)
    return mat
def lap_mat(n_rows, n_col):
    # D block diag aka laplacian block
    D_mat = scipy.sparse.lil_matrix((n_col, n_col))
    D_mat = set_D(D_mat)

    # -I block with offset
    maim_mat = scipy.sparse.block_diag([D_mat] * n_rows).tolil()
    maim_mat = set_minos_I(maim_mat, n_col)

    return maim_mat

def pedding(img, new_width, new_height, center, isMask):
    black = (0, 0, 0)
    if isMask:
        old_height, old_width = img.shape
        result = np.full((new_height, new_width), 0, dtype=np.uint8)
    else:
        old_height, old_width, channels = img.shape
        result = np.full((new_height, new_width, channels), black, dtype=np.uint8)

    offset_row = center[1] - old_height // 2
    offset_col = center[0] - old_width // 2

    result[offset_row:offset_row + old_height, offset_col:offset_col + old_width] = img
    return result

def set_outside_rows_mat(mat, mask):
    rows, cols = mask.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if mask[i, j] == 0:
                k = j + i * cols
                mat[k, k] = 1
                mat[k, k + 1] = 0
                mat[k, k - 1] = 0
                mat[k, k + cols] = 0
                mat[k, k - cols] = 0
    return mat

def set_outside_rows_b(b, mask, im_tgt):
    for i in range(b.shape[0]):
        # if this pixel is outside
        if mask[i] == 0:
            b[i] = im_tgt[i]
    return b

def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    new_height, new_width, channels = im_tgt.shape
    im_src = pedding(im_src, new_width, new_height, center, False)
    im_mask = pedding(im_mask, new_width, new_height, center, True)
    laplac = lap_mat(new_height, new_width)  # square matrix n_rows * n_col

    # we want I(x,y) = T(x,y) where we are outside the source
    A_mat = set_outside_rows_mat(laplac, im_mask)
    A_mat = A_mat.tocsc()

    for channel in range(3):
        im_trg_flat = im_tgt[:new_height, :new_width, channel].flatten()
        im_src_flat = im_src[:new_height, :new_width, channel].flatten()
        im_mask_flat = im_mask.flatten()
        b = laplac.dot(im_src_flat)
        b = set_outside_rows_b(b, im_mask_flat, im_trg_flat)
        x = spsolve(A_mat, b)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.reshape((new_height, new_width)).astype('uint8')
        im_tgt[:new_height, :new_width, channel] = x

    im_blend = im_tgt
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
