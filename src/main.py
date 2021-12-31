import torch
import os
import wandb
import time
import datetime

import pytorch_lightning as pl
from .training import DeepVONet, CTCNet
from .utils import plot_test
from .model import ConvLstmNet


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
    )

    if args["CTC"]:
        model = CTCNet(args)
    else:
        model = DeepVONet(args)
    
    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model)

    trainer.fit(model)
    print("------- Training Done! -------")

    print("------- Testing Begins! -------")
    trainer.test(model)
    plot_test(model.test_data, model.trajectories)
    save_model_onnx(model, args)


def save_model_onnx(model, args):
    # convert model to onnx

    # set the model to inference mode
    model.eval()
    torch.save(model.architecture.state_dict(), args["model_name"])
    wandb.save(args["model_name"])

    model.to('cpu')

    if type(model.architecture) is ConvLstmNet:
        model.architecture.reset_hidden_states(bsize=args["bsize"], zero=True, cpu=True)  # reset to 0 the hidden states of RNN

    x = torch.randn(args["bsize"], 6, args["resize"] // 2, args["resize"], requires_grad=False)

    filename = os.path.join(wandb.run.dir,
                            f"{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M')}_bestmodel.onnx")

    # Export the model
    torch.onnx.export(model.architecture,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}},
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    torch.save(model.architecture.state_dict(), args["model_name"])
