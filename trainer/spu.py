


import os
from tqdm import tqdm

import torch
import torch.nn as nn

from trainer.finetune import Finetune


def zerolike_params_dict(model, device=None):
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    """

    return [
        (k, torch.zeros_like(p).to(p.device if (device == None) else device))
        for k, p in model.named_parameters()
    ]


def copy_params_dict(model, copy_grad=False, device=None):
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """

    if copy_grad:
        return [(k, p.grad.data.detach().clone()) for k, p in model.named_parameters()]
    else:
        return [(k, p.data.detach().clone().to(p.device if (device == None) else device)) for k, p in
                model.named_parameters()]


class SPU(Finetune):
    def __init__(self, args):
        super().__init__(args)
        self.mask = {}
        self.trainable_params = []
        

    def compute_score(self, model, dataloader, **kwargs):


        # Initialize importance matrix
        importance = dict(zerolike_params_dict(model, device=self.args.device))

        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        model.zero_grad()
        total_batch =  len(dataloader)
        num_batch_for_score = total_batch * self.args.score_batch_percentage
        print(f'Total batch for importance {total_batch}, use {num_batch_for_score} batches')
        stop_flag = 1


        for num_batch, batch in enumerate(tqdm(dataloader)):
            stop_flag = 1
            # Get batch
            images, _, texts = batch
            images = images.to(self.args.device)
            texts = texts.to(self.args.device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Average L2-Norm of the output
            loss = torch.norm(logits_per_image, p="fro", dim=1).pow(2).mean()
            loss.backward()

            # Accumulate importance
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        importance[name].data += param.grad.clone()
                        if importance[name].data.abs().min() < 1e-12:
                            stop_flag = 0
            if num_batch > num_batch_for_score and stop_flag:
                break

        return importance

        
    def compute_importance(self, dataset, model, task):

            
        cur_set = dataset.get_dataset(task, is_train=True, with_buffer=False)
        loader = self.get_loader(cur_set, is_train=True)
        print('Compute importance for the current task...')
        cur_importance = self.compute_score(model, loader, task=task,dataset=dataset)
        if self.args.save_ckpt:
            with open(os.path.join(self.args.log_path, f'task{task}_importance.torchSave'), 'wb') as file:
                torch.save(cur_importance, file)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.trainable_params:
                    if any(param in name for param in self.args.full_update_param) or self.args.selection_rate == 1.0:
                        self.mask[name] = torch.ones(param.shape, dtype=param.dtype).to(self.args.device)
                        continue
                    if name not in cur_importance.keys():
                        print(f' importance of `{name} is none')
                        continue
                    importance = cur_importance[name]
                    magnitudes = importance.abs()
                    k = int(magnitudes.numel() * self.args.selection_rate)
   
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    self.mask[name] = torch.zeros_like(magnitudes).to(self.args.device)
                    self.mask[name].view(-1)[topk_indices] = 1
                    
    def unfreeze_model(self, model):
        model.train()
        for name, param in model.named_parameters():
            if name == 'visual.proj':
                if self.args.finetune_proj:
                    trainable_params = True
                else:
                    trainable_params = False
            else:
                trainable_params = self.args.edit_layer in name
                
            if trainable_params:
                param.requires_grad = True
                if name not in self.trainable_params:
                    self.trainable_params.append(name)
            else:
                param.requires_grad = False
        print('Trainable parameters: ', self.trainable_params)
        
    def update_model(self, model, optimizer, **kwargs):
        with torch.no_grad():
            for name, param in model.named_parameters():
                gradients = param.grad
                if gradients is not None:
                    param.grad = self.mask[name] * param.grad
        optimizer.step()

