
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


class MAS(Finetune):
    def __init__(self, args):
        super().__init__(args)
        self.magnitudes = {}
        self.alpha = 0.5
        self._lambda = self.args.scale
        self.importance_computed = False

        self.trainable_params = []
        
    def setup_importance(self, model):
        # Parameters before the first task starts
        self.params = dict(copy_params_dict(model))
        # Initialize Fisher information weight importance
        self.importance = dict(zerolike_params_dict(model))
        
    def compute_update_mas_importance(self, model, dataloader):
        self.params = dict(copy_params_dict(model), device=self.args.device)

        # Get importance
        curr_importance = self._get_importance(model, dataloader)
        if not self.importance_computed:
            self.importance = curr_importance
            self.importance_computed = True
            return
        else:
            # Update importance
            for name in self.importance.keys():
                self.importance[name] = (self.alpha * self.importance[name]
                                         + (1 - self.alpha) * curr_importance[name].data)
    
    def _get_importance(self, model, dataloader):


        # Initialize importance matrix
        importance = dict(zerolike_params_dict(model, device=self.args.device))

        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        size = 0
        
        for num_batch, batch in enumerate(tqdm(dataloader)):
            # Get batch
            images, _, texts = batch
            images = images.to(self.args.device)
            texts = texts.to(self.args.device)

            # Forward pass
            model.zero_grad()
            logits_per_image, logits_per_text = model(images, texts)

            # Average L2-Norm of the output
            loss = torch.norm(logits_per_image, p="fro", dim=1).pow(2).mean()

            loss.backward()

            # Accumulate importance
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        importance[name].data += param.grad.clone().abs() * len(images)
            size += len(images)
            

        # Normalize importance
        importance = {
            name: importance[name] / size
            for name in importance.keys()
        }
        if self.args.importance_max_normalize:
            for name in importance.keys():
                if name in self.trainable_params:
                    importance[name] = torch.nn.functional.normalize(importance[name],dim=-1,p=torch.inf)

        return importance

        
    def compute_importance(self, dataset, model, task):
        if task == 0:
            self.setup_importance(model)
        else:
            prev_set = dataset.get_dataset(task - 1, is_train=True, with_buffer=False)
            buffer = self.get_loader(prev_set)
            print ('Compute importance for the last task...')
            self.compute_update_mas_importance(model, buffer)
    
    def compute_loss(self, batch, model, **kwargs):
        buffer = kwargs.get('buffer', None)
        epoch = kwargs.get('epoch', 0)
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        images, _, texts = batch
        if buffer and epoch > 0:
            images_b, _, texts_b = buffer
            images = torch.cat([images, images_b])
            texts = torch.cat([texts, texts_b])

        images = images.to(self.args.device)
        texts = texts.to(self.args.device)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=self.args.device)

        logits_per_image, logits_per_text = model(images, texts)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        if self.args.scale > 0.0:
            penalty = self._lambda * self.compute_importance_penalty(self.args, model)
            return total_loss + penalty
        else:
            return total_loss
        
    def compute_importance_penalty(self, args, model):
        loss_reg = torch.tensor(0).float().to(self.args.device)

        # Apply penalty term
        for name, param in model.named_parameters():

            if name in self.trainable_params:
                loss_reg += torch.sum(
                    self.importance[name] *
                    (param - self.params[name].expand(param.shape)).pow(2)
                )

        # Update loss
        return loss_reg

        
    
    