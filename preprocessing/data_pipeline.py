from torch.utils.data import DataLoader
from preprocessing.preprocess_ucr import UCRDataset, AugUCRDataset, UCRDatasetImporter
from preprocessing.augmentations import Augmenter

"""
def build_data_pipeline(batch_size, dataset_importer: UCRDatasetImporter, config: dict, kind: str, n_pairs=2, shuffle_train=True) -> DataLoader:
    
    num_workers = config['dataset']["num_workers"]

    if not augment:
        # DataLoader
        if kind == 'train':
            train_dataset = UCRDataset("train", dataset_importer)
            return DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=shuffle_train, drop_last=False, pin_memory=True)  # `drop_last=False` due to some datasets with a very small dataset size.
        elif kind == 'test':
            test_dataset = UCRDataset("test", dataset_importer)
            return DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)
        else:
            raise ValueError
    else:
        augmenter = Augmenter(**config['augmentations'])
        # DataLoader
        if kind == 'train':
            train_dataset = AugUCRDataset("train", dataset_importer, augmenter, n_pairs=n_pairs)
            return DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=shuffle_train, drop_last=False, pin_memory=True)  # `drop_last=False` due to some datasets with a very small dataset size.
        elif kind == 'test':
            test_dataset = UCRDataset("test", dataset_importer)
            return DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)
        else:
            raise ValueError
"""


def build_data_pipeline(
    batch_size: int,
    dataset_importer: UCRDatasetImporter,
    config: dict,
    kind: str,
    augment: bool = False,
    n_pairs: int = 2,
    shuffle_train: bool = True,
) -> DataLoader:
    """
    :param config:
    :param kind: train/valid/test
    :param augment: Whether to apply data augmentation.
    :param n_pairs: Number of pairs to create when augmenting data (if applicable).
    :param shuffle_train: Whether to shuffle the training data.
    :param augment: Whether to apply data augmentation.
    """
    num_workers = config["dataset"]["num_workers"]

    if kind not in ["train", "test"]:
        raise ValueError("Kind must be 'train' or 'test'")

    if augment:
        augmenter = Augmenter(**config["augmentations"])
        dataset = AugUCRDataset(kind, dataset_importer, augmenter, n_pairs=n_pairs)
    else:
        dataset = UCRDataset(kind, dataset_importer)

    shuffle = shuffle_train if kind == "train" else False

    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
    )
