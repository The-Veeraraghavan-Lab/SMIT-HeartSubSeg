# Data

This repo expects the dataset tree at `data/AllDatasets`.

Create `data/AllDatasets` as a local symlink to the real centralized dataset store.

Expected layout:

```text
data/
├── README.md
└── AllDatasets/
    ├── heartsub_master.json
    ├── HeartSubv2_substructs/
    │   ├── image/
    │   ├── label_plus/
    │   ├── label_plus_peri/        # if used
    │   └── dose/
    ├── Breast66/
    │   ├── imgs/
    │   ├── label_plus/
    │   ├── label_plus_peri/        # if used
    │   └── dose/
```

## Dataset JSON format

`heartsub_master.json` is a MONAI-style datalist. Each split is a list of entries with `image` and `label` paths relative to `AllDatasets/`:

```json
{
  "set1_cnc64_training": [
    {
      "image": "HeartSubv2_substructs/image/case001.nii.gz",
      "label": "HeartSubv2_substructs/label_plus/case001_label.nii.gz"
    }
  ],
  "breast66_labeled": [
    {
      "image": "Breast66/imgs/case001.nii.gz",
      "label_plus": "Breast66/label_plus/case001_label.nii.gz"
    }
  ]
}
```

Multi-label NIfTI files use integer indices matching the channel order described in the main README.

Do not commit dataset contents, manifests, labels, predictions, or other sensitive files into this folder.
