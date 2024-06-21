import time
import numpy as np
from typing import Tuple


import torch
import torch.optim as optim
from torchvision import transforms

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from trainer.finetune import Finetune
from metric import AverageMeter
from trainer.utils import logging


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


def prune_weight(w, prune_ratio):
    inputs = w.clone().detach().abs()
    k = int(inputs.numel() * (1-prune_ratio))
    if w.grad is not None:
        inputs += w.grad.clone().detach().abs()
    topk_values, _ = torch.topk(inputs.view(-1), k=k)

    threshold = topk_values.min()
    outputs = torch.ones_like(inputs)
    outputs[inputs.abs() < threshold] = 0

    return outputs


def weight_growing(mask,  lower_bound_value, upper_bound_value):
    entries = mask.numel()
    num_non_zero = mask.sum()

    num_added_zeros = int((entries - num_non_zero) - upper_bound_value * entries)
    num_added_zeros = num_added_zeros if num_added_zeros > 0 else 0

    false_indices = (mask == 0).nonzero(as_tuple=False)
    selected_indices = false_indices[torch.randperm(false_indices.size(0))[:num_added_zeros]]
    mask[selected_indices[:, 0], selected_indices[:, 1]] = 1

    return mask


class SparseCL(Finetune):
    def __init__(self, args):
        super().__init__(args)
        self.sp_mask_update_freq = args.sp_mask_update_freq
        self.masked_layers = []

        self.upper_bound = args.upper_bound
        self.lower_bound = args.lower_bound
        self.mask_update_decay_epoch = self.args.mask_update_decay_epoch

        

    def mask_init(self, model):
        self.masks = dict(zerolike_params_dict(model))
        for name, _ in model.named_parameters():
            if 'weight' in name and 'ln' not in name and 'conv' not in name and 'embed' not in name:
                self.masked_layers.append(name)

    def update_mask(self, model, epoch):
        for ee, ll, uu in zip(self.mask_update_decay_epoch, self.lower_bound, self.upper_bound):
            lower_bound_value = ll
            upper_bound_value = uu
            if epoch < ee:
                break

        freq = self.sp_mask_update_freq
        if epoch % freq == 0:
            with torch.no_grad():

                for name, weight in (model.named_parameters()):
                    if name not in self.masked_layers:
                        continue

                    num_nonzeros = torch.count_nonzero(self.masks[name])
                    total_num = self.masks[name].numel()
                    sparsity = (num_nonzeros * 1.0) / total_num
                    # print(("\n==> BEFORE UPDATE: {}: {}, {}, {}".format(name,str(num_nonzeros),str(total_num),str(sparsity))))

                    ############## pruning #############
                    self.masks[name] = prune_weight(weight.clone().detach(), lower_bound_value)
                    num_nonzeros = torch.count_nonzero(self.masks[name])
                    total_num = self.masks[name].numel()
                    sparsity = (num_nonzeros * 1.0) / total_num
                    # print(("\n==> AFTER PRUNE: {}: {}, {}, {}".format(name,str(num_nonzeros),str(total_num),str(sparsity))))

                    ############## growing #############
                    self.masks[name] = weight_growing(self.masks[name],
                                                      lower_bound_value,
                                                      upper_bound_value)
                    num_nonzeros = torch.count_nonzero(self.masks[name])
                    total_num = self.masks[name].numel()
                    sparsity = (num_nonzeros * 1.0) / total_num
                    # print(("\n==> AFTER GROW: {}: {}, {}, {}".format(name,str(num_nonzeros),str(total_num),str(sparsity))))

    def unfreeze_model(self, model):
        model.train()
        for name, param in model.named_parameters():
            if name in self.masked_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_iterator(self, dataset, task):
        trainset = dataset.get_dataset(
            task, is_train=True, with_buffer=False)
        print(trainset)
        train_dataloader = self.get_loader(trainset, is_train=True)
        total_batches = len(train_dataloader)

        return train_dataloader, total_batches

    def update_model(self, model, optimizer, **kwargs):
        with torch.no_grad():
            for name, param in model.named_parameters():
                gradients = param.grad
                if gradients is not None:
                    param.grad = self.masks[name] * param.grad

        optimizer.step()

    def train(self, model, dataset, task):
        if task == 0:
            self.buffer = Buffer(dataset.buffer_size, 'cpu')
            self.mask_init(model)
        train_dataloader, total_batches = self.get_iterator(
            dataset, task)

        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=0.2)

        if  self.args.scheduler:
            self.lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=self.args.epochs * 10,
                cycle_mult=1.0,
                max_lr=self.args.lr,
                min_lr=0,
                warmup_steps=1
            )
        self.unfreeze_model(model)
        batch_time = AverageMeter()
        loss = AverageMeter()

        optimizer.zero_grad()
        for epoch in range(self.args.epochs):
            self.update_mask(model, epoch)
            for iiter, batch in enumerate(train_dataloader):
                batch_size = self.get_batch_size(batch)
                end = time.time()
                if task > 0 and not self.buffer.is_empty():
                    buf_inputs, buf_labels = self.buffer.get_data(batch_size)
                    buffer = [buf_inputs, None, buf_labels]
                else:
                    buffer = None

                total_loss = self.compute_loss(
                    batch, model, epoch=epoch, buffer=buffer)
                total_loss.backward()

                self.update_model(model, optimizer,
                                  count=batch_size, epoch=epoch, task=task)

                optimizer.zero_grad()
                self.buffer.add_data(examples=batch[0], labels=batch[-1])

                batch_time.update(time.time() - end)
                loss.update(total_loss.item() / batch_size, n=batch_size)
                logging('iter', iiter + epoch * total_batches,
                        f'train_loss/{task}', loss.val, self.args)
                if iiter % self.args.print_frequency == 0:
                    print(' Epoch: [{0}/{1}], Batch: [{2}/{3}]\t'.format(epoch, self.args.epochs, iiter, total_batches),
                          f'Batch Time {batch_time.val: .3f} ({batch_time.avg: .3f})\t'
                          f'Loss {loss.val:.4f} ({loss.avg: .4f}) \t'
                          f'Estimated Task Time {batch_time.avg * total_batches * self.args.epochs / 3600: .3f} H'
                          )

            if  self.args.scheduler:
                self.lr_scheduler.step()

        model.eval()

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size

class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32

                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms = None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
