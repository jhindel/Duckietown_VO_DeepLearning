import torch
from itertools import product as cartProduct
import pandas as pd
import os
import wandb
import time
import datetime

from .dataset import DuckietownDataset
from .utils import human_format, count_parameters
from .model import DeepVONet
from .training import plot_test, plot_train_valid, train_model, test_model


def training_testing(args, wandb_project, visualization=True, wandb_name=None):
    # tell wandb to get started
    run = wandb.init(project=wandb_project, entity="av_deepvo", name=wandb_name, config=args)
    # access all HPs through wandb.config, so logging matches execution!
    # wandb.config = args
    print(args)
    train_data = DuckietownDataset(args["train_split"], args)
    val_data = DuckietownDataset(args["val_split"], args)
    test_data = DuckietownDataset(args["test_split"], args)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args["bsize"], shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args["bsize"], shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)

    if not os.path.exists(args["checkpoint_path"]): os.makedirs(args["checkpoint_path"])

    model = DeepVONet(args["resize"], args["resize"], args["dropout_p"])

    print("Number of parameters:", human_format(count_parameters(model)))
    print("Number of parameter bytes:", human_format(32 * count_parameters(model)))

    USE_GPU = torch.cuda.is_available()

    if USE_GPU:
        model.cuda()
        print("Running model with cuda")

    if args["checkpoint"] is not None:
        the_checkpoint = torch.load(args["checkpoint"])
        model.load_state_dict(the_checkpoint['state_dict'])

    best_model, logs, args_ = train_model(model, train_loader, val_loader, args)
    if visualization:
        plot_train_valid(logs, args_)

    print(logs)

    test_loss, relative_poses_pred = test_model(best_model, test_loader, args)
    if visualization:
        plot_test(test_data, relative_poses_pred)

    save_model_onnx(best_model, args)

    run.finish()

    return logs, test_loss.detach().numpy()


def hyperparamter_tuning(args, wandb_project, visualization=False, wandb_name=None):
    hyper_parameter_combinations = list(
        cartProduct(*[args[param] for param in args.keys()]))
    hyper_parameter_set_list = [dict(zip(args.keys(), hyper_parameter_combinations[i])) for i in
                                range(len(hyper_parameter_combinations))]

    evaluation_overview = pd.DataFrame(columns=list(args.keys()) + ['train_loss', 'val_loss', 'test_loss'])
    for i, hyper_parameter in enumerate(hyper_parameter_set_list):
        print('%s/%s:  %s' % (i, len(hyper_parameter_set_list), hyper_parameter))
        results, test_loss = training_testing(hyper_parameter, wandb_project, visualization=visualization, wandb_name=f"{wandb_name}_{i}")
        hyper_parameter.update({'train_loss': results['train_loss'][-1], 'val_loss': results['val_loss'][-1],
                                'test_loss': test_loss})
        evaluation_overview = evaluation_overview.append(hyper_parameter, ignore_index=True)
    evaluation_overview.to_csv('model_evaluation_all.csv')


def save_model_onnx(model, args):
    # convert model to onnx

    # set the model to inference mode
    model.eval()
    model.reset_hidden_states(args["bsize"], zero=True, cpu=True)
    model.to('cpu')

    x = torch.randn(args["bsize"], 3, args["resize"], args["resize"], requires_grad=False)

    filename = os.path.join(args["checkpoint_path"],
                            f"{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M')}_bestmodel.onnx")

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    wandb.save(filename)