
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from preprocessing.preprocess_ucr import UCRDatasetImporter
    from preprocessing.preprocess_ucr import UCRDataset
    from preprocessing.preprocess_ucr import AugUCRDataset
    from preprocessing.data_pipeline import build_data_pipeline
    from utils import load_yaml_param_settings

    config_dir = 'configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['stage1']
    train_data_loader_non_aug, test_data_loader= [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    train_data_loader_aug = build_data_pipeline(batch_size, dataset_importer, config, augment=True, kind="train")

    for b in train_data_loader_aug:
        (x1, x2), y = b
        break

    x1 = x1.unsqueeze(1)
    x2 = x2.unsqueeze(1)

    i = 10
    plt.plot(x1[i])
    plt.plot(x2[i])