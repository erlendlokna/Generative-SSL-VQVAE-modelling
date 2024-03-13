from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import (
    load_yaml_param_settings,
    get_root_dir,
    model_filename,
)


from trainers.train_maskgit import train_maskgit
from trainers.train_byol_maskgit import train_byol_maskgit
from trainers.train_mage import train_mage

UCR_SUBSET = [
    # "ElectricDevices",
    # "StarLightCurves",
    # "Wafer",
    # "ECG5000",
    # "TwoPatterns",
    "FordA",
    "UWaveGestureLibraryAll",
    # "FordB",
    # "ChlorineConcentration",
    # "ShapesAll",
]

FINISHED_STAGE1 = {}
FINISHED_STAGE2 = {}

STAGE1_EPOCHS = 1500
STAGE2_EPOCHS = 3000

NUM_RUNS_PER = 1

STAGE1_METHODS = [""]
STAGE2_METHODS = ["vibcreg"]  # "vibcreg"]

SSL_WEIGHTS = {"barlowtwins": 1.0, "vicreg": 0.01, "vibcreg": 0.001, "": 0}


def run_experiments():
    config_dir = get_root_dir().joinpath("configs", "config.yaml")
    config = load_yaml_param_settings(config_dir)
    config["trainer_params"]["max_epochs"]["stage1"] = STAGE1_EPOCHS
    config["trainer_params"]["max_epochs"]["stage2"] = STAGE2_EPOCHS
    batch_size = config["dataset"]["batch_sizes"]["stage1"]

    project_name_stage1 = "SSL_VQVAE-STAGE1-IDUN"
    project_name_stage2 = "SSL_VQVAE-STAGE2-IDUN"

    for dataset in UCR_SUBSET:
        c = config.copy()
        c["SSL"]["stage1_method"] = ""
        c["SSL"]["stage2_method"] = ""
        c["dataset"]["dataset_name"] = dataset

        dataset_importer = UCRDatasetImporter(**config["dataset"])

        train_data_loader_no_aug = build_data_pipeline(
            batch_size, dataset_importer, config, augment=False, kind="train"
        )

        test_data_loader = build_data_pipeline(
            batch_size, dataset_importer, config, augment=False, kind="test"
        )

        # STAGE 2
        for method_1 in STAGE1_METHODS:
            c["SSL"]["stage1_method"] = method_1
            c["SSL"]["stage1_weight"] = SSL_WEIGHTS[method_1]
            # c["VQVAE"]["decorrelate_codebook"] = False

            for method_2 in STAGE2_METHODS:
                c["SSL"]["stage2_method"] = method_2
                c["SSL"]["stage2_weight"] = SSL_WEIGHTS[method_2]
                if method_2 == "":
                    for run in range(NUM_RUNS_PER):
                        train_maskgit(
                            config=c,
                            train_data_loader=train_data_loader_no_aug,
                            test_data_loader=test_data_loader,
                            do_validate=True,
                            gpu_device_idx=0,
                            wandb_run_name=f"{model_filename(c, 'maskgit')}-{dataset}-run{run+1}",
                            wandb_project_name=project_name_stage2,
                        )
                else:
                    for run in range(NUM_RUNS_PER):
                        train_byol_maskgit(
                            config=c,
                            train_data_loader=train_data_loader_no_aug,
                            test_data_loader=test_data_loader,
                            do_validate=True,
                            gpu_device_idx=0,
                            wandb_run_name=f"{model_filename(c, 'byolmaskgit')}-{dataset}-run{run+1}",
                            wandb_project_name=project_name_stage2,
                        )


if __name__ == "__main__":
    run_experiments()
