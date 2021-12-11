import torch
from itertools import product as cartProduct
import pandas as pd
import os

from src.dataset import DuckietownDataset


from src.utils import human_format, count_parameters
from src.model import DeepVONet
from src.training import plot_test, plot_train_valid, train_model, test_model


def training_testing(args, visualization=True):

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

    if args["checkpoint"] is not None:
        the_checkpoint = torch.load(args["checkpoint"])
        model.load_state_dict(the_checkpoint['state_dict'])

    best_model, logs, args_ = train_model(model, train_loader, val_loader, args)
    if visualization:
        plot_train_valid(logs, args_)

    test_loss, predicted_test_relative_poses = test_model(best_model, test_loader, args)
    if visualization:
        plot_test(test_data, predicted_test_relative_poses, args)

    # TODO figure out what to return
    return logs


def hyperparamter_tuning(args):
    hyper_parameter_combinations = list(
        cartProduct(*[args[param] for param in args.keys()]))
    hyper_parameter_set_list = [dict(zip(args.keys(), hyper_parameter_combinations[i])) for i in
                                range(len(hyper_parameter_combinations))]

    evaluation_overview = pd.DataFrame(columns=list(args.keys()) + ['val_loss', 'loss'])
    for i, hyper_parameter in enumerate(hyper_parameter_set_list):
        print('%s/%s:  %s' % (i, len(hyper_parameter_set_list), hyper_parameter))
        results = training_testing(hyper_parameter, visualization=False)
        hyper_parameter.update({'val_loss': results['loss'][-1], 'loss': results['loss'][-1]})
        evaluation_overview = evaluation_overview.append(hyper_parameter, ignore_index=True)
    evaluation_overview.to_csv('model_evaluation_all.csv')