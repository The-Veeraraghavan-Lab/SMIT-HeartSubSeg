import os
import os.path as osp
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import nibabel as nib
from natsort import natsorted

from dosimetrics_utils import get_voxel_cc, get_max_dose, get_vcc, get_mean_dose

RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results'
ALLDATASETS_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'AllDatasets'

def compute_dosimetrics(evaldata, dosedir, folder_op, output_dir=RESULTS_ROOT / 'xcelrecords_dose'):
    """Compute dosimetrics for a given evaluation dataset."""
    
    columns = ['name', 'aorta', 'pa', 'pv', 'svc', 'ivc', 'ra', 'rv', 'la', 'lv']
    df = pd.DataFrame(columns=columns)
    
    filelist = natsorted(os.listdir(folder_op))
    
    for filename in tqdm(filelist, desc=evaldata):
        predlabel = nib.load(osp.join(folder_op, filename)).get_fdata()
        
        filename = filename.replace("_label","")
        
        gtdose_filename = osp.join(dosedir, filename.replace(".nii.gz", "_dose.nii.gz"))
        gtdose_nib = nib.load(gtdose_filename)
        vcc = get_voxel_cc(gtdose_nib.affine)
        gtdose = gtdose_nib.get_fdata()
        
        doses = {i: gtdose[predlabel == i] for i in range(1, 10)}
        
        df = df._append({
            'name': filename.replace(".nii.gz", ""),
            'aorta': get_max_dose(doses[1]),
            'pa': get_vcc(doses[2], 40, vcc),
            'pv': get_max_dose(doses[3]),
            'svc': get_max_dose(doses[4]),
            'ivc': get_max_dose(doses[5]),
            'ra': get_mean_dose(doses[6]),
            'rv': get_mean_dose(doses[7]),
            'la': get_mean_dose(doses[8]),
            'lv': get_mean_dose(doses[9])
        }, ignore_index=True)
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(osp.join(output_dir, f'{evaldata}.csv'), index=False)
    print(f"Saved: {evaldata}.csv")


# Dataset configurations
CONFIGS = {
    'oar': {
        'dosedir': str(ALLDATASETS_ROOT / 'HeartSubv2_substructs' / 'dose'),
        'gt_label_dir': str(ALLDATASETS_ROOT / 'HeartSubv2_substructs' / 'label_plus'),
        'model_outputs': ['run1_plus_cnc64_bnorm'],
        'model_folder_template': str(RESULTS_ROOT / 'model_outputs_lung' / '{}')
    },
    'breast66': {
        'dosedir': str(ALLDATASETS_ROOT / 'Breast66' / 'dose'),
        'gt_label_dir': str(ALLDATASETS_ROOT / 'Breast66' / 'label_plus'),
        'model_outputs': ['run1_plus_cnc64_bnorm'],
        'model_folder_template': str(RESULTS_ROOT / 'model_outputs_breast' / '{}')
    }
}

if __name__ == '__main__':
    # for mode, config in CONFIGS.items():
    for mode in ['breast66']:
        config = CONFIGS[mode]

        # Ground truth
        compute_dosimetrics(
            evaldata='gt',
            dosedir=config['dosedir'],
            folder_op=config['gt_label_dir'],
            output_dir=RESULTS_ROOT / 'xcelrecords_dose' / mode
        )
        
        # Model predictions
        for model_name in config['model_outputs']:
            compute_dosimetrics(
                evaldata=model_name,
                dosedir=config['dosedir'],
                folder_op=config['model_folder_template'].format(model_name),
                output_dir=RESULTS_ROOT / 'xcelrecords_dose' / mode
            )