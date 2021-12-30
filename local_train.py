import wandb

from src.dataset_split import train, val, test, test_dummy, train_dummy, val_dummy
from src.main import training_testing


if __name__ == '__main__':
    wandb.login()

    args ={"data_dir":"/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/data/",
           "model":"ConvNet", "train_split": train_dummy, "val_split": val_dummy, "test_split": test_dummy,
        "checkpoint_path":'./checkpoint', "checkpoint":None, "bsize":32, "lr":0.001,
        "weight_decay":1e-4, "trajectory_length":5, "dropout_p":0.5,
        "resize":64, "K":100, "epochs":1, "patience":40}

    training_testing(args, wandb_project="deepvo-general-trials-JH", wandb_name="trial_runs")


