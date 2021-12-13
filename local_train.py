import wandb

from src.dataset_split import train, val, test, test_dummy, train_dummy, val_dummy
from src.main import hyperparamter_tuning, training_testing


wandb.login()

args ={"data_dir":("/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/data/",),
    "train_split":(train_dummy,), "val_split":(val_dummy,), "test_split":(test_dummy,),
    "checkpoint_path":('./checkpoint',), "checkpoint":(None,), "bsize":(2,), "lr":(0.001,),
    "weight_decay":(1e-4,), "trajectory_length":(5,), "dropout_p":(0.85,),
    "resize":(64,), "K":(100,), "epochs":(1,), "patience":(40,), "camera-correction":(True,)}

# hyperparamter_tuning(args, wandb_project="deepvo-finetuning-trial", visualization=True, wandb_name="trial_runs")

args ={"data_dir":"/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/data/",
    "train_split":train_dummy, "val_split":val_dummy, "test_split":test_dummy,
    "checkpoint_path":'./checkpoint', "checkpoint":None, "bsize":2, "lr":0.001, 
    "weight_decay":1e-4, "trajectory_length":5, "dropout_p":0.85,
    "resize":64, "K":100, "epochs":1, "patience":40, "camera-correction":True}

training_testing(args, wandb_project="deepvo-finetuning-trial", visualization=True, wandb_name="trial_runs")


