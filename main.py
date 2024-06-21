import hydra
import omegaconf
from argparse import Namespace

import os

import time
from datetime import datetime


import random
import numpy as np

import torch


from clip import clip
from utils import * 
from dataset import DATASET
from trainer import METHOD

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(args):
    args = omegaconf.OmegaConf.to_container(args)
    args = Namespace(**args)
    
    random_seed(args.seed)

    start = time.time()

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # --------------------------------
    # setup logging 
    # --------------------------------
    if args.name is None:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = '-'.join([
            date_str,
            args.method,
            args.dataset,
        ])
    log_base_path = os.path.join(args.logs, args.name)
    os.makedirs(log_base_path, exist_ok=True)
    args.log_path = log_base_path
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.name, config=vars(args), dir=log_base_path)

    print("{}".format(args).replace(', ', ',\n'))



    # --------------------------------
    # setup model
    # --------------------------------
    if args.model_source == 'clip':
        model, transform = clip.load(args.model, download_root=args.model_root,args=args)
    elif args.model_source == 'hf-lora':
        model = prepare_hf_lora_model(args)
        transform = clip._transform(args.input)
    else:
        raise ValueError
    

    # --------------------------------
    # setup data
    # --------------------------------
    dataset = DATASET[args.dataset](args, args.data, transform)
    args.num_tasks = dataset.num_tasks
    args.scenario = dataset.scenario

    # --------------------------------
    # setup method and start training
    # --------------------------------
    Trainer = METHOD[args.method](args)
    for task in range(args.num_tasks):
        if args.debug and task == 3:
            break
        print (f"Start task {task}")
        if args.evaluation:
            Trainer.only_evaluation(model, dataset, task)
            continue
        Trainer.train(model, dataset, task)
        if not args.no_eval:
            Trainer.evaluation(model, dataset, task)
            Trainer.save_checkpoint(model, task, args)
            
    print(f'Total training time in hours: {(time.time() - start) / 3600: .3f}')
    if args.wandb:
        wandb.finish()




if __name__ == '__main__':
    main()
