from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import  DataLoader

from peft.tuners.lora import LoraLayer
from peft.utils import transpose

from clip.clip import tokenize
from utils import prepare_hf_lora_model
from trainer.finetune import Finetune
from trainer.utils import accuracy, logging
from dataset.cc12m import cc12m
from dataset.imagenet import ImageNet
from metric import AverageMeter

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
        return [(k, p.grad.data.detach().clone()) for k, p in model.named_parameters() if p.requires_grad]
    else:
        return [(k, p.data.detach().clone().to(p.device if (device == None) else device)) for k, p in
                model.named_parameters() if p.requires_grad]

class LoRAEWC(Finetune):
    def __init__(self, args):
        super().__init__(args)
        self.alpha = 0.95
        self.ewc_lambda = self.args.scale
    def setup_importance(self, model):
        # Parameters before the first task starts
        self.params = dict(copy_params_dict(model))
        # Initialize Fisher information weight importance
        self.importance = dict(zerolike_params_dict(model))
        
    def compute_importance(self, dataset, model, task):

        if task == 0:
            self.trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]


            pretrained_model = prepare_hf_lora_model(self.args, peft=False)
            self.setup_importance(pretrained_model)
            self.compute_fisher(dataset, pretrained_model)
            del pretrained_model
            
    def compute_fisher(self, dataset, model):

        print('Compute importance for conditional set...')
        condset = cc12m(transform=dataset.transform,root=self.args.cc12m_root)
        dataloader = self.get_loader(condset)
        self.params = dict(copy_params_dict(model), device=self.args.device)

        # Get importance
        self.importance = self._get_importance(model, dataloader)
        
    def _get_importance(self, model, dataloader,  loss_type='l2'):

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
            images = images.to(self.args.device)
            texts = texts.to(self.args.device)

            ground_truth = torch.arange(
                len(images), dtype=torch.long, device=self.args.device)

            outputs = model(texts, images)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            ground_truth = torch.arange(len(images), dtype=torch.long, device=self.args.device)
            loss_img = nn.CrossEntropyLoss()
            loss_txt = nn.CrossEntropyLoss()
            loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            loss.backward()

            # Accumulate importance
            for name, param in model.named_parameters():
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
                    importance[name] = torch.nn.functional.normalize(importance[name], dim=-1, p=torch.inf)

        return importance
    
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

        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=self.args.device)

        outputs = model(texts, images)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        ce_loss = (loss_img(logits_per_image, ground_truth) +
                   loss_txt(logits_per_text, ground_truth)) / 2

        # EWC losss
        ewc_loss = 0
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):


                if module.active_adapter not in module.lora_A.keys():
                    continue


                fan_in_fan_out = module.fan_in_fan_out
                adapter_weights = transpose(
                    module.lora_B[module.active_adapter].weight @ module.lora_A[module.active_adapter].weight,
                    fan_in_fan_out,
                ) * module.scaling[module.active_adapter]

                name = name.replace('base_model.model.', '') + '.weight'
                fisher_matrix_weights = self.importance[name]
                ewc_loss += torch.sum(fisher_matrix_weights * (adapter_weights ** 2))

        loss = ce_loss + self.ewc_lambda * ewc_loss

        return loss
    
    def eva_task_t(self, t, dataset, testset, model, task):
        zero_shot_metric = AverageMeter()
        avg_metric = AverageMeter()
        with torch.no_grad():

            text_inputs_full = torch.cat(
                [tokenize(f"a photo of a {c}") for c in dataset.class_name_full]).cuda()

        if task < dataset.num_tasks - 1:
            unseen_class_idx = torch.Tensor(np.concatenate(dataset.task_classes[task + 1:], axis=None)).to(torch.long)
            text_inputs_seen = text_inputs_full.clone().detach()
            text_inputs_seen[unseen_class_idx] = 0
        else:
            text_inputs_seen = text_inputs_full.clone().detach()
        

        test_dataloader = DataLoader(
            testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        for (image, label, _) in tqdm(test_dataloader, desc=f"Evaluation for {t}",
                                      total=len(test_dataloader)):
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():

                if t <= task:  # update average accuracy for current batch
                    logits = model(text_inputs_seen, image).logits_per_image
                    acc = accuracy(logits, label)[0]
                    avg_metric.update(acc, image.size(0))

                # update zero-shot accuracy for current batch
                logits_full = model(text_inputs_full, image).logits_per_image
                acc_full = accuracy(logits_full, label)[0]
                zero_shot_metric.update(acc_full, image.size(0))

        avg = avg_metric.avg if not torch.is_tensor(avg_metric.avg) else avg_metric.avg.item()
        unseen_avg = zero_shot_metric.avg if not torch.is_tensor(
            zero_shot_metric.avg) else zero_shot_metric.avg.item()

        return avg, unseen_avg, len(testset)

    def held_out_evaluation(self, model, transform):
        testset = ImageNet(transform=transform,root=self.args.imagenet_root)
        metric = AverageMeter()
        names = testset.classes
        with torch.no_grad():

            text_inputs_full = torch.cat(
                [tokenize(f"a photo of a {c}") for c in names]).cuda()
        test_dataloader = DataLoader(
            testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        for image, label in tqdm(test_dataloader, desc=f"Evaluation for ImageNet Validation Set",
                                 total=len(test_dataloader)):
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():

                logits = model(text_inputs_full, image).logits_per_image
            acc = accuracy(logits, label)[0]
            metric.update(acc, image.size(0))
        return metric.avg.item()
    def middle_evaluation(self, model, dataset, task, epoch, validset=None, log_name=None):
        pass

    def evaluation(self, model, dataset, task, log=True):
        # Currently this is only for class-incremental learning

        unseen_metric = self.unseen_metric
        avg_metric = self.metric

        for t in range(self.args.num_tasks):
            testset = dataset.get_dataset(t, is_train=False)
            acc, acc_full, n = self.eva_task_t(
                t, dataset, testset, model, task)

            # update for current task
            self.full_metric.update(task, t, acc_full, n=n)
            self.full_metric.update_metric(task, t)
            if t <= task:
                avg_metric.update(task, t, acc, n=n)
                avg_metric.update_metric(task, t)
            else:
                unseen_metric.update(task, t, acc_full, n=n)
                unseen_metric.update_metric(task, t)
            if self.args.report_to:
                logging('task', task, f'{t}/accuracy per task', acc, self.args)

        held_out = self.held_out_evaluation(model, dataset.transform) if not (
            self.args.debug ) else 0
        self.held_out_metric.update(held_out)

        if not log:
            return avg_metric.average_accuracy[task], unseen_metric.average_accuracy[task]

        print(
            f' * End evaluation: task accuracy top1 {self.metric.average_accuracy[task]:.2f}')
        print(
            f' * End evaluation: forgetting top1 {self.metric.forgetting[task]:.2f}')
        print(
            f' * End evaluation: learning top1 {self.metric.learning[task]:.2f}')
        print(
            f' * End evaluation: average learning top1 {self.metric.learning[:task+1].mean():.2f}')
        print(
            f' * End evaluation: unseen accuracy top1 {self.unseen_metric.average_accuracy[task]:.2f}')
        print(
            f' * End evaluation: whole set evaluation top1 {self.full_metric.average_accuracy[task]:.2f}')
        print(f'* End evaluation: held out top1 {held_out:.2f}')

        if self.args.report_to:
            logging('task', task, 'average accuracy',
                    self.metric.average_accuracy[task], self.args)
            logging('task', task, 'forgetting',
                    self.metric.forgetting[task], self.args)
            logging('task', task, 'learning',
                    self.metric.learning[task], self.args)
            logging('task', task, 'average learning',
                    self.metric.learning[:task+1].mean(), self.args)
            logging('task', task, 'unseen accuracy',
                    self.unseen_metric.average_accuracy[task], self.args)
            logging('task', task, 'held out accuracy', held_out, self.args)
            logging('task', task, 'full set accuracy', self.full_metric.average_accuracy[task], self.args)