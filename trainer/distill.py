
import copy
import time

import torch
import torch.nn as nn

from trainer.finetune import Finetune
from trainer.utils import logging
from metric import AverageMeter


def kl_loc_loss(pre, post, r):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    pre_ = pre.view(-1, pre.shape[-1]) / r
    post_ = post.view(pre_.shape) / r
    assert pre_.shape[0] == post_.shape[0]

    kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum()

    return kl


def ps_distill_loss(pre, post, r=2.0):
    # input image feature, do transpose first
    pre = pre.to(torch.float32).t() / r
    post = post.to(torch.float32).t() / r
    # each row of pre,post corresponds to a text feature's (normalized) similarities to current batch image features
    q = pre.softmax(-1)
    log_p = post.log_softmax(-1)
    loss = torch.sum(-q * log_p, dim=-1).mean()
    return loss


class Distillation(Finetune):
    def compute_loss(self, batch, model, **kwargs):

        task = kwargs.get('task', 0)
        distill_model = kwargs.get('distill_model', None)
        distill_loss_type = kwargs.get('distill_loss_type', 'visual')

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        if distill_loss_type == 'visual':
            dloss = kl_loc_loss
        elif distill_loss_type == 'text':
            dloss = ps_distill_loss
        else:
            raise NotImplementedError

        if task == 0:
            images, label, texts = batch
            images = images.to(self.args.device)
            texts = texts.to(self.args.device)
            distill_loss = 0
        else:
            (images, label, texts), (images_prev, _, texts_prev) = batch
            images = images.to(self.args.device)
            texts = texts.to(self.args.device)
            images_prev = images_prev.to(self.args.device)
            texts_prev = texts_prev.to(self.args.device)
            images_prev = torch.cat([images_prev,images])
            texts_prev = torch.cat([texts_prev,texts])

            with torch.no_grad():
                image_features_prev_distill, _ = distill_model(images_prev, texts_prev)
            image_features_prev, _ = model(images_prev, texts_prev)
            distill_loss = dloss(image_features_prev, image_features_prev_distill, r=self.args.tem)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=self.args.device)
        logits_per_image, logits_per_text = model(images, texts)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,
                                                                          ground_truth)) / 2

        if task > 0:
            total_loss += self.args.scale * distill_loss
        return total_loss, distill_loss


    def train(self, model, dataset, task):
        train_dataloader, buffer_loader, total_batches = self.get_iterator(
            dataset, task)
        
        optimizer = self.get_optimizer(model)
        
        distill_model = copy.deepcopy(model)
        distill_model.eval()
        for param in distill_model.parameters():
            param.requires_grad = False
        
        self.unfreeze_model(model)
        
        batch_time = AverageMeter()
        loss = AverageMeter()
        dloss = AverageMeter()
        optimizer.zero_grad()
        self.compute_importance(dataset, model, task=task)
        optimizer.zero_grad()
        for epoch in range(self.args.epochs):
            buffer_iterator = iter(buffer_loader) if buffer_loader else None
            for iiter, batch in enumerate(train_dataloader):
                batch_size = self.get_batch_size(batch)
                end = time.time()

                if buffer_iterator:
                    try:
                        batch_b = next(buffer_iterator)
                    except StopIteration:
                        buffer_iterator = iter(buffer_loader)
                        batch_b = next(buffer_iterator)
                    batch = [batch, batch_b]

                total_loss, distill_loss = self.compute_loss(batch, model, task=task, distill_model=distill_model,
                                                             distill_loss_type=self.args.distill_loss)
                total_loss.backward()

                self.update_model(model, optimizer,
                                  count=batch_size, epoch=epoch, task=task)


                optimizer.zero_grad()

                batch_time.update(time.time() - end)
                loss.update(total_loss.item() / batch_size, n=batch_size)
                dloss.update(total_loss.item() / batch_size, n=batch_size)
                logging('iter', iiter + epoch * total_batches, f'train_loss/{task}_distill', loss.val, self.args)
                logging('iter', iiter + epoch * total_batches, f'train_loss/{task}', loss.val, self.args)
                if iiter % self.args.print_frequency == 0:
                    print(' Epoch: [{0}/{1}], Batch: [{2}/{3}]\t'.format(epoch, self.args.epochs, iiter,
                                                                         total_batches),
                          f'Batch Time {batch_time.val: .3f} ({batch_time.avg: .3f})\t'
                          f'Loss {loss.val:.4f} ({loss.avg: .4f}) \t'
                          f'Estimated Task Time {batch_time.avg * total_batches * self.args.epochs / 3600: .3f} H'
                          )
            if self.args.scheduler:
                self.lr_scheduler.step()
        model.eval()
        print('Update Buffer....')
        dataset.update_buffer(task)
        