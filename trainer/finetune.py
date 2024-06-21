import time

import torch
import torch.nn as nn
import torch.optim as optim

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from trainer.frozenclip import FrozenCLIP
from metric import AverageMeter
from trainer.utils import accuracy, get_ckpt_save_path, logging

class Finetune(FrozenCLIP):
    def get_loader(self, dataset, is_train=False):
        if dataset is None:
            return None

        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers, sampler=None, drop_last=is_train )
        return train_dataloader
    
    def get_iterator(self, dataset, task):
        if self.args.balanced_buffer and task > 0:
            trainset = dataset.get_dataset(
                task, is_train=True, with_buffer=False)
            bufferset = dataset.get_buffer(task) if task > 0 else None
            print('buffer:', bufferset)

        else:
            trainset = dataset.get_dataset(
                task, is_train=True, with_buffer=(self.args.buffer_size > 0 and task > 0))
            bufferset = None

        if bufferset:
            buffer_loader = self.get_loader(bufferset)
        else:
            buffer_loader = None
        train_dataloader = self.get_loader(trainset, is_train=True)
        total_batches = len(train_dataloader)

        return train_dataloader, buffer_loader, total_batches

    def get_optimizer(self, model):
        network_params = []
        for n, p in model.named_parameters():
            if 'transformer' in n or 'ViT' in self.args.model:
                network_params.append({'params': p, 'lr': self.args.lr})
            else:
                network_params.append({'params': p, 'lr': self.args.lr_resnet})


        optimizer = optim.AdamW(network_params, lr=self.args.lr, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=self.args.wd)
        if self.args.scheduler: 
            self.lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=self.args.epochs * self.args.num_tasks,
                cycle_mult=1.0,
                max_lr=self.args.lr,
                min_lr=0,
                warmup_steps=1
            )
        return optimizer
    def unfreeze_model(self, model):
        model.train()
        
    def compute_importance(self, dataset, model, **kwargs):
        pass
    
    def get_batch_size(self, batch, **kwargs):
        return batch[0].size(0)
    
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

        logits_per_image, logits_per_text = model(images, texts)

        total_loss = (loss_img(logits_per_image, ground_truth) +
                      loss_txt(logits_per_text, ground_truth)) / 2
        return total_loss
    
    def update_model(self, model, optimizer, **kwargs):
        optimizer.step()


    def train(self, model, dataset, task):
        train_dataloader, buffer_loader, total_batches = self.get_iterator(
            dataset, task)
        
        optimizer = self.get_optimizer(model)
        
        self.unfreeze_model(model)
        
        batch_time = AverageMeter()
        loss = AverageMeter()
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
                else:
                    batch_b = None

                total_loss = self.compute_loss(
                    batch, model, buffer=batch_b, epoch=epoch)
                total_loss.backward()

                self.update_model(model, optimizer,
                                  count=batch_size, epoch=epoch, task=task)


                optimizer.zero_grad()

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
            if self.args.scheduler:
                self.lr_scheduler.step()
        model.eval()
        print('Update Buffer....')
        dataset.update_buffer(task)

        
        