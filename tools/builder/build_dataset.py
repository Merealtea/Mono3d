from datasets import *

datasets = {
    'CustomMonoDataset': CustomMonoDataset,
}


def build_dataset(cfg, data_root, annotation_prefix, image_prefix, eval):
    if not eval:
       pipeline = cfg["train_pipeline"]
    else:
       pipeline = cfg["test_pipeline"]

    if cfg['dataset_type'] in datasets:
        dataset = CustomMonoDataset(data_root, annotation_prefix, image_prefix, pipeline, test_mode=eval)
    else:
        dataset = None
    return dataset