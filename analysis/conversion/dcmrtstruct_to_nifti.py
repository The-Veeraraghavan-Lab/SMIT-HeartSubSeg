#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a folder of DICOM + RTSTRUCT cases into NIfTI.

Update the input and output roots below before running.
"""

import os
import os.path as osp
from dcmrtstruct2nii import dcmrtstruct2nii

input_root = 'data/rtstruct_input'
output_root = 'data/rtstruct_nifti'

folders = os.listdir(input_root)
folders = [folder for folder in folders if not folder.endswith('.DS_Store')]

for folder in folders:
    path_dicom = osp.join(input_root, folder, 'DCM')

    root_rtstruct = osp.join(input_root, folder, 'RTSTRUCT')
    path_rtstruct = osp.join(root_rtstruct, os.listdir(root_rtstruct)[0])

    dcmrtstruct2nii(path_rtstruct, path_dicom, osp.join(output_root, folder))
