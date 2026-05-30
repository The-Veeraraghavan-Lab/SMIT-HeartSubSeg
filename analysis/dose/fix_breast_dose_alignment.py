"""
Interactive Dose Map Registration Fixer
Allows flipping dose maps in x, y, z directions to match the original image.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import os.path as osp
from pathlib import Path

ALLDATASETS_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'AllDatasets'


class DoseMapFixer:
    def __init__(self, image_dir, dose_dir, output_dir=None):
        self.image_dir = image_dir
        self.dose_dir = dose_dir
        self.output_dir = output_dir or dose_dir + '_fixed'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.image = None
        self.dose = None
        self.dose_fixed = None
        self.image_nib = None
        self.current_id = None
        self.flip_state = {'x': False, 'y': False, 'z': False}
        self.rot_state = {'xy': 0, 'xz': 0, 'yz': 0}  # number of 90° rotations
    
    def load_case(self, case_id):
        self.current_id = case_id
        self.flip_state = {'x': False, 'y': False, 'z': False}
        self.rot_state = {'xy': 0, 'xz': 0, 'yz': 0}
        
        # Find image file
        image_path = osp.join(self.image_dir, f'{case_id}.nii.gz')
        if not osp.exists(image_path):
            image_path = osp.join(self.image_dir, f'{case_id}_image.nii.gz')
        if not osp.exists(image_path):
            raise FileNotFoundError(f"Image not found for {case_id}")
        
        # Find dose file
        dose_path = osp.join(self.dose_dir, f'{case_id}_dose.nii.gz')
        if not osp.exists(dose_path):
            dose_path = osp.join(self.dose_dir, f'{case_id}.nii.gz')
        if not osp.exists(dose_path):
            raise FileNotFoundError(f"Dose map not found for {case_id}")
        
        print(f"Loading image: {image_path}")
        print(f"Loading dose:  {dose_path}")
        
        self.image_nib = nib.load(image_path)
        self.image = self.image_nib.get_fdata()
        self.dose = nib.load(dose_path).get_fdata()
        self.dose_fixed = self.dose.copy()
        
        print(f"Image shape: {self.image.shape}")
        print(f"Dose shape:  {self.dose.shape}")
        
        if self.image.shape != self.dose.shape:
            print("WARNING: Shape mismatch! May need resampling.")
        
        self.show_comparison()
    
    def flip(self, axis):
        """
        Flip the dose map along specified axis.
        Args:
            axis: 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz', or 'reset'
        """
        if axis == 'reset':
            self.dose_fixed = self.dose.copy()
            self.flip_state = {'x': False, 'y': False, 'z': False}
            self.rot_state = {'xy': 0, 'xz': 0, 'yz': 0}
            print("Reset to original")
        else:
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            for a in axis:
                if a in axis_map:
                    self.dose_fixed = np.flip(self.dose_fixed, axis=axis_map[a])
                    self.flip_state[a] = not self.flip_state[a]
                    print(f"Flipped along {a}-axis")
        
        self._print_state()
        self.show_comparison()
    
    def rot(self, plane, degrees=90):
        """
        Rotate the dose map in specified plane.
        Args:
            plane: 'xy' (axial), 'xz' (coronal), 'yz' (sagittal)
            degrees: 90, 180, or 270
        """
        plane_map = {
            'xy': (0, 1),  # rotate in axial plane
            'xz': (0, 2),  # rotate in coronal plane
            'yz': (1, 2),  # rotate in sagittal plane
        }
        
        if plane not in plane_map:
            print(f"Invalid plane: {plane}. Use 'xy', 'xz', or 'yz'")
            return
        
        if degrees not in [90, 180, 270]:
            print(f"Invalid degrees: {degrees}. Use 90, 180, or 270")
            return
        
        n_rots = degrees // 90
        axes = plane_map[plane]
        
        self.dose_fixed = np.rot90(self.dose_fixed, k=n_rots, axes=axes)
        self.rot_state[plane] = (self.rot_state[plane] + n_rots) % 4
        
        print(f"Rotated {degrees}° in {plane} plane")
        self._print_state()
        self.show_comparison()
    
    def _print_state(self):
        print(f"Flip state: {self.flip_state}")
        print(f"Rotation state (x90°): {self.rot_state}")
        
    def show_comparison(self, slice_idx=None):
        """Show old vs new overlay comparison."""
        if slice_idx is None:
            slice_idx = self.image.shape[2] // 2
        slice_idx = 72
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        z_idx = slice_idx
        x_idx = self.image.shape[0] // 2
        y_idx = self.image.shape[1] // 2
        
        # Row 1: Original overlay (Axial, Sagittal, Coronal)
        axes[0, 0].imshow(self.image[:, :, z_idx].T, cmap='gray', origin='lower')
        axes[0, 0].imshow(self.dose[:, :, z_idx].T, cmap='hot', alpha=0.5, origin='lower')
        axes[0, 0].set_title('Original Overlay (Axial)')
        
        axes[0, 1].imshow(self.image[x_idx, :, :].T, cmap='gray', origin='lower', aspect='auto')
        axes[0, 1].imshow(self.dose[x_idx, :, :].T, cmap='hot', alpha=0.5, origin='lower', aspect='auto')
        axes[0, 1].set_title('Original Overlay (Sagittal)')
        
        axes[0, 2].imshow(self.image[:, y_idx, :].T, cmap='gray', origin='lower', aspect='auto')
        axes[0, 2].imshow(self.dose[:, y_idx, :].T, cmap='hot', alpha=0.5, origin='lower', aspect='auto')
        axes[0, 2].set_title('Original Overlay (Coronal)')
        
        # Row 2: Fixed overlay (Axial, Sagittal, Coronal)
        axes[1, 0].imshow(self.image[:, :, z_idx].T, cmap='gray', origin='lower')
        axes[1, 0].imshow(self.dose_fixed[:, :, z_idx].T, cmap='hot', alpha=0.5, origin='lower')
        axes[1, 0].set_title(f'Fixed Overlay (Axial) - {self.flip_state}')
        
        axes[1, 1].imshow(self.image[x_idx, :, :].T, cmap='gray', origin='lower', aspect='auto')
        axes[1, 1].imshow(self.dose_fixed[x_idx, :, :].T, cmap='hot', alpha=0.5, origin='lower', aspect='auto')
        axes[1, 1].set_title('Fixed Overlay (Sagittal)')
        
        axes[1, 2].imshow(self.image[:, y_idx, :].T, cmap='gray', origin='lower', aspect='auto')
        axes[1, 2].imshow(self.dose_fixed[:, y_idx, :].T, cmap='hot', alpha=0.5, origin='lower', aspect='auto')
        axes[1, 2].set_title('Fixed Overlay (Coronal)')
        
        for ax in axes.flat:
            ax.axis('off')
        
        plt.suptitle(f'Case: {self.current_id}', fontsize=14)
        plt.tight_layout()
    
    def browse_slices(self, view='axial'):
        """Interactive slice browser."""
        from matplotlib.widgets import Slider
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if view == 'axial':
            max_slice = self.image.shape[2] - 1
            get_slices = lambda idx: (
                self.image[:, :, idx].T,
                self.dose_fixed[:, :, idx].T
            )
        elif view == 'sagittal':
            max_slice = self.image.shape[0] - 1
            get_slices = lambda idx: (
                self.image[idx, :, :].T,
                self.dose_fixed[idx, :, :].T
            )
        else:  # coronal
            max_slice = self.image.shape[1] - 1
            get_slices = lambda idx: (
                self.image[:, idx, :].T,
                self.dose_fixed[:, idx, :].T
            )
        
        init_idx = max_slice // 2
        img_slice, dose_slice = get_slices(init_idx)
        
        im0 = axes[0].imshow(img_slice, cmap='gray', origin='lower')
        axes[0].set_title('Image')
        
        im1 = axes[1].imshow(dose_slice, cmap='hot', origin='lower')
        axes[1].set_title('Dose Fixed')
        
        im2 = axes[2].imshow(img_slice, cmap='gray', origin='lower')
        im2_overlay = axes[2].imshow(dose_slice, cmap='hot', alpha=0.5, origin='lower')
        axes[2].set_title('Overlay')
        
        for ax in axes:
            ax.axis('off')
        
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        slider = Slider(ax_slider, 'Slice', 0, max_slice, valinit=init_idx, valstep=1)
        
        def update(val):
            idx = int(slider.val)
            img_slice, dose_slice = get_slices(idx)
            im0.set_data(img_slice)
            im1.set_data(dose_slice)
            im2.set_data(img_slice)
            im2_overlay.set_data(dose_slice)
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        plt.show()
    
    def save(self, suffix='_fixed'):
        """Save the fixed dose map using the image's affine and header."""
        output_path = osp.join(self.output_dir, f'{self.current_id}_dose{suffix}.nii.gz')
        
        # Create new NIfTI with image's affine
        fixed_nib = nib.Nifti1Image(self.dose_fixed, self.image_nib.affine, self.image_nib.header)
        nib.save(fixed_nib, output_path)
        
        print(f"Saved: {output_path}")
        print(f"Applied flips: {self.flip_state}")
        return output_path


def interactive_fix(case_id, image_dir, dose_dir, output_dir=None):
    """Convenience function for interactive fixing."""
    fixer = DoseMapFixer(image_dir, dose_dir, output_dir)
    fixer.load_case(case_id)
    
    print("\n" + "="*60)
    print("COMMANDS:")
    print("  fixer.flip('x')      - Flip along x-axis")
    print("  fixer.flip('y')      - Flip along y-axis")
    print("  fixer.flip('z')      - Flip along z-axis")
    print("  fixer.flip('xy')     - Flip along x and y")
    print("  fixer.flip('reset')  - Reset to original")
    print("  fixer.rot('xy')      - Rotate 90° in axial plane")
    print("  fixer.rot('xz')      - Rotate 90° in coronal plane")
    print("  fixer.rot('yz')      - Rotate 90° in sagittal plane")
    print("  fixer.rot('xy2')     - Rotate 180° in axial plane")
    print("  fixer.rot('xy3')     - Rotate 270° in axial plane")
    print("  fixer.show_comparison(slice_idx)")
    print("  fixer.save()")
    print("="*60 + "\n")
    
    return fixer

if __name__ == "__main__":
    # Example usage
    fixer = interactive_fix(
        case_id = 'XXXX', # example case
        image_dir=str(ALLDATASETS_ROOT / 'Breast66' / 'imgs'),
        dose_dir=str(ALLDATASETS_ROOT / 'Breast66' / 'dose'),
        output_dir=str(ALLDATASETS_ROOT / 'Breast66' / 'dose')
    )
    
    # Then interactively:
    # fixer.flip('x')
    # fixer.rot('xy', 270)
    fixer.flip('z')

    # fixer.flip('xzy')
    # fixer.save()
