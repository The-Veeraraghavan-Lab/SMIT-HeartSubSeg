import numpy as np


def get_voxel_cc(affine):
    voxel_dims_mm = np.abs(affine[:3, :3].diagonal())
    voxel_volume_mm3 = np.prod(voxel_dims_mm)
    voxel_volume_cc = voxel_volume_mm3 / 1000
    return voxel_volume_cc

def get_max_dose(arr):
    if len(arr) > 0:
        return np.nanmax(arr)
    else:
        return 0

def get_dcc(arr, d = 10, vxcc = 0):
    if len(arr) > 0:
        vxdose_sorted = sorted(arr, reverse = True)
        num_voxels_10cc = int(d/vxcc)
        d10cc = vxdose_sorted[num_voxels_10cc - 1]
        return d10cc
    else:
        return 0
    
def get_vcc(arr, d = 10, vxcc = 0):
    if len(arr) > 0:
        dv = np.sum(arr > d)
        return ((dv/vxcc)/(len(arr)/vxcc)*100)
    else:
        return 0
    
def get_mean_dose(arr):
    if len(arr) > 0:
        return np.nanmean(arr)
    else:
        return 0
    
    
# def get_dcc_vec(arr, d = 10, vxcc = 0):
#     if len(arr) > 0:
#         vxdose_sorted = sorted(arr, reverse = True)
#         num_voxels_10cc = int(d/vxcc)
#         d10cc_vec = vxdose_sorted[:num_voxels_10cc - 1]
#         return d10cc_vec
#     else:
#         return 0