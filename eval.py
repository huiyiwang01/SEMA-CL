import sys
import logging
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import re
import numpy as np

def eval(args):
    args["seed"]=args["seed"][0]
    device = copy.deepcopy(args["device"])

    logs_name = "logs/{}/{}".format(args["model_name"],args["backbone_type"])
    logfilename = "logs/{}/{}/eval_{}".format(
        args["model_name"],
        args["backbone_type"],
        args["dataset"],
    )
        
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    model._cur_task = data_manager.nb_tasks-1
    model._network.fc = nn.Linear(768, data_manager.nb_classes)
    model._total_classes = data_manager.nb_classes
    test_dataset = data_manager.get_dataset(np.arange(0, model._total_classes), source="test", mode="test" )
    model.test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=8)
    adapter_pattern = get_adapter_pattern(args["checkpt_path"])
    for idx, n in enumerate(adapter_pattern):
        if n > 1:
            for i in range(n-1):
                model._network.backbone.blocks[idx].adapter_module.add_adapter()
                model._network.backbone.blocks[idx].adapter_module.end_of_task_training()
    model.load_checkpoint(args["checkpt_path"])
    model._network.to(args["device"][0])
    cnn_accy, _ = model.eval_task()
    logging.info("CNN: {}".format(cnn_accy["grouped"]))
    

def get_adapter_pattern(checkpt_path):
    state_dict = torch.load(checkpt_path)
    adapter_pattern = [1]*12
    pattern = re.compile(r'backbone\.blocks\.(\d+)\.adapter_module\.adapters\.(\d+)\.')

    for key in state_dict.keys():
        match = pattern.search(key)
        if match:
            block_id = int(match.group(1))
            adapter_id = int(match.group(2))
            adapter_pattern[block_id] = max(adapter_pattern[block_id], adapter_id+1)
    return adapter_pattern


    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))