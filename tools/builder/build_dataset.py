from datasets import *

datasets = {
    'CustomMonoDataset': CustomMonoDataset,
    'CustomMV3DDataset': CustomMV3DDataset
}


def build_dataset(cfg):
    if not cfg['test_mode']:
       pipeline = cfg["train_pipeline"]
    else:
       pipeline = cfg["test_pipeline"]

    cfg['pipeline'] = pipeline

    if cfg['dataset_type'] in datasets:
        dataset = datasets[cfg['dataset_type']](**cfg)
    else:
        dataset = None
        raise ValueError(f"Unrecognized dataset type: {cfg['dataset_type']}")
    return dataset