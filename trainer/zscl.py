import copy
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from dataset.cc12m import cc12m
from metric import AverageMeter
from trainer.finetune import Finetune
from trainer.utils import logging

def distillation(t, s, T=2):
    p = F.softmax(t / T, dim=1)
    loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
    return loss

def merge_we(model_0, model_1, sma_count):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
    return model_1

class ZSCL(Finetune):
    
    def compute_loss(self, batch, model, **kwargs):
        buffer = kwargs.get('buffer', None)
        epoch = kwargs.get('epoch', 0)
        ref_batch = kwargs.get('ref_batch', None)
        ref_model = kwargs.get('ref_model', None)
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
        
        assert ref_batch is not None
        ################ ZSCL change
        ref_images, _, ref_texts = ref_batch
        ref_images = ref_images.to(self.args.device)
        ref_texts = ref_texts.to(self.args.device)
        with torch.no_grad():
            ref_embeddings = ref_model.encode_text(ref_texts)
            ref_embeddings = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)
            ref_out = ref_model.encode_image(ref_images)
            ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)
        
        ref_out_current = model.encode_image(ref_images)
        ref_out_current = ref_out_current / ref_out_current.norm(dim=-1, keepdim=True)
        
        logit_scale = model.logit_scale.exp()
        logits_current = logit_scale * ref_out_current @ ref_embeddings.t()
        logits_ref = logit_scale * ref_out @ ref_embeddings.t()
        loss_visual_distill = distillation(logits_ref,logits_current,T=2)

        logits_current_2 = logits_current.t()
        logits_ref_2 = logits_ref.t()
        loss_logits_distill = distillation(logits_ref_2,logits_current_2,T=2)
        total_loss = total_loss + self.args.distill_scale* (loss_visual_distill +  loss_logits_distill)
        
        return total_loss
    
    def train(self, model, dataset, task):
        train_dataloader, buffer_loader, total_batches = self.get_iterator(
            dataset, task)
        
        optimizer = self.get_optimizer(model)
        
        
        self.unfreeze_model(model)
        batch_time = AverageMeter()
        loss = AverageMeter()

        optimizer.zero_grad()
        
        ################ ZSCL change
        we_model = copy.deepcopy(model)
        we_model.cuda()
        for param in we_model.parameters():
            param.requires_grad = False
        
        ref_model , _  = clip.load(self.args.model, download_root=self.args.model_root, device='cuda', jit=False)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        
        ref_dataset = cc12m(transform=dataset.transform, root=self.args.cc12m_root)
        ref_dataloader = self.get_loader(ref_dataset)
        ref_iter = iter(ref_dataloader)
        
        ############### ZSCL change ends
        

        we_n = 0
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
                    
                try:
                    ref_batch = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(ref_dataloader)
                    ref_batch = next(ref_iter) 

                total_loss = self.compute_loss(
                    batch, model, buffer=batch_b, epoch=epoch,ref_batch=ref_batch,ref_model=ref_model)
                total_loss.backward()

                self.update_model(model, optimizer,
                                  count=batch_size, epoch=epoch, task=task)

                if iiter % self.args.avg_freq == 0:
                    we_n += 1
                    merge_we(model,we_model,we_n)
                
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
        
