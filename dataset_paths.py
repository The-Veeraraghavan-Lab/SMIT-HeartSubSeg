import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / 'data' / 'AllDatasets'


def get_default_data_root():
    return str(DEFAULT_DATA_ROOT)


def resolve_dataset_json(data_root, dataset_json):
    if os.path.isabs(dataset_json):
        return dataset_json
    return os.path.join(data_root, dataset_json)
