import torch
import torch.nn.functional as F
import numpy as np
from numpy import array, cos, sin, pi
import nibabel as nib


def rotate(img: torch.Tensor, mat: torch.Tensor):
    """
    rotate a given tensor by specific angles based on rotation matrix.
    No pure translation is applied by default in this implementation.
    This may lead to part of feature being outbound from the FOV, so do padding properly in your own work.
    :param img: tensor to be rotated with shape
    (B, C, Nx, Ny, Nz) in 5-D.
    :param mat: rotation matrix in pythonic format, aka the flipped matlab rotation matrix with shape
    (B, C, 3, 3) in 4-D.
    :return: the rotated tensor.

    """
    _, c, w, h, d = img.shape
    max_dim = max(w, h, d)

    # cubic padding to avoid unreasonably stretching
    img = F.pad(img, [(max_dim - d) // 2, (max_dim - d) // 2, (max_dim - h) // 2,
                                                     (max_dim - h) // 2, (max_dim - w) // 2, (max_dim - w) // 2])
    # add no pure translation
    pure_translation = torch.zeros(1, 3, 1).to(img.device)

    affine_matrix = torch.cat([mat.squeeze(0), pure_translation], dim=2)
    grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)
    
    return F.grid_sample(input=img, grid=grid, mode='bilinear')


def skew(vector):

    """
    skew-symmetric operator for rotation matrix generation
    """

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def get_rotation_mat(ori1: np.ndarray, ori2: np.ndarray):
    """
    generating pythonic style rotation matrix
    :param ori1: your current orientation
    :param ori2: orientation to be rotated
    :return: pythonic rotation matrix.
    """
    v = np.cross(ori1, ori2)
    c = np.dot(ori1, ori2)
    mat = np.identity(3) + skew(v) + np.matmul(skew(v), skew(v)) / (1 + c)
    return np.flip(mat).copy()

# code for test


def generate_cube(canvas_size, cube_size=100):
    cube = np.ones([cube_size, cube_size, cube_size])
    padding = (canvas_size - cube_size) // 2
    cube = np.pad(cube, ((padding, padding), (padding, padding), (padding, padding)))
    return cube


if __name__ == '__main__':

    mat = get_rotation_mat(array([0,0,1]), 
                        array([cos(pi / 3) * sin(pi / 5), sin(pi / 3) * sin(pi / 5), cos(pi / 5)]))
    print(mat)
    inv_mat = mat.T
    mat = torch.from_numpy(mat[np.newaxis, np.newaxis]).float()
    cube = torch.from_numpy(generate_cube(256)[np.newaxis, np.newaxis]).float()
    chi_rot1 = rotate(cube, mat)
    nib.save(nib.Nifti1Image(chi_rot1.squeeze().numpy(), np.eye(4)), 'chi_rot.nii')