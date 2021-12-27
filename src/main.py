import torch
from itertools import product as cartProduct
import pandas as pd
import os
import wandb
import time
import datetime
import matplotlib.pyplot as plt
import sys

import pytorch_lightning as pl
from .training import DeepVONet
from .utils import plot_test


def training_testing(args, wandb_project, wandb_name=None):
    # experiment tracker (you need to sign in with your account)

    wandb.require(experiment="service")

    wandb_logger = pl.loggers.WandbLogger(
        name=wandb_name,
        log_model=True,  # save best model using checkpoint callback
        project=wandb_project,
        entity="av_deepvo",
        config=args,
    )

    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["EXP_LOG_DIR"] = wandb_logger.experiment.dir

    # to save the best model on validation
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="best_model" + str(time.time()),
        monitor="valid_loss",
        save_top_k=1,
        mode="max",
        save_last=False,
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        devices=1, accelerator="auto", strategy="dp",
        logger=wandb_logger,
        callbacks=checkpoint,
        max_epochs=args["epochs"],
        # log_every_n_steps=50,
        # progress_bar_refresh_rate=2
    )

    model = DeepVONet(args)
    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model)

    trainer.fit(model)
    print("------- Training Done! -------")

    print("------- Testing Begins! -------")
    trainer.test(model)
    plot_test(model.test_data, model.trajectories)
    save_model_onnx(model, args)

    # wandb_logger.unwatch(model)


def hyperparamter_tuning(args, wandb_project, wandb_name=None):
    hyper_parameter_combinations = list(
        cartProduct(*[args[param] for param in args.keys()]))
    hyper_parameter_set_list = [dict(zip(args.keys(), hyper_parameter_combinations[i])) for i in
                                range(len(hyper_parameter_combinations))]

    evaluation_overview = pd.DataFrame(columns=list(args.keys()) + ['train_loss', 'val_loss', 'test_loss'])
    for i, hyper_parameter in enumerate(hyper_parameter_set_list):
        print('%s/%s:  %s' % (i, len(hyper_parameter_set_list), hyper_parameter))
        results, test_loss = training_testing(hyper_parameter, wandb_project, wandb_name=f"{wandb_name}_{i}")
        hyper_parameter.update({'train_loss': results['train_loss'][-1], 'val_loss': results['val_loss'][-1],
                                'test_loss': test_loss})
        evaluation_overview = evaluation_overview.append(hyper_parameter, ignore_index=True)
    evaluation_overview.to_csv('model_evaluation_all.csv')


def save_model_onnx(model, args):
    # convert model to onnx

    # set the model to inference mode
    model.eval()
    # model.reset_hidden_states(args["bsize"], zero=True, cpu=True)
    model.to('cpu')

    x = torch.randn(args["bsize"], 6, args["resize"], args["resize"], requires_grad=False)

    filename = os.path.join(wandb.run.dir,
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

    # os.path.join(wandb.run.dir, "model.h5")
    # wandb.save(filename)
