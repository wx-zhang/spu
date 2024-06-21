
import time

import numpy as np

import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from clip.clip import tokenize
from trainer.finetune import Finetune
from trainer.utils import logging
from metric import AverageMeter


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


class SLCA(Finetune):
    def get_optimizer(self, network_params, epochs):

        optimizer = optim.AdamW(network_params, lr=self.args.lr, betas=(
            0.9, 0.999), eps=1e-8, weight_decay=0.2)

        if self.args.scheduler:
            self.lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=epochs,
                cycle_mult=1.0,
                max_lr=self.args.lr,
                min_lr=0,
                warmup_steps=1
            )
        return optimizer

    def train(self, model, dataset, task):

        network_params = []
        count = 0
        for n, p in model.named_parameters():

            if n in ['visual.proj', 'text_projection']:
                count += 1
                network_params.append({'params': p, 'lr': self.args.lr*self.args.head_lr_scale})
            else:
                network_params.append({'params': p, 'lr': self.args.lr})
        self.train_phase1(model, dataset, task, network_params)

        self._compute_class_mean(model, dataset, task, check_diff=False)

        network_params = []
        for n, p in model.named_parameters():

            if n in ['visual.proj', 'text_projection']:
                count += 1
                network_params.append({'params': p, 'lr': self.args.lr})
        self.train_phase2(model, dataset, task, network_params)

        print('Update Buffer....')
        dataset.update_buffer(task)

        model.eval()

    def train_phase1(self, model, dataset, task, network_params):
        epochs = self.args.epochs
        train_dataloader, buffer_loader, total_batches = self.get_iterator(
            dataset, task)
        optimizer = self.get_optimizer(network_params, epochs)

        # check(model,self.args)
        self.unfreeze_model(model)
        batch_time = AverageMeter()
        loss = AverageMeter()

        optimizer.zero_grad()
        for epoch in range(epochs):
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

    def _compute_class_mean(self, model, dataset, task, check_diff=False):
        train_dataloader, buffer_loader, total_batches = self.get_iterator(
            dataset, task)
        feature_dim = model.visual.proj.shape[0]
        if not hasattr(self, '_class_means'):
            self._class_means = torch.zeros((dataset.num_classes, feature_dim))
            self._class_covs = torch.zeros((dataset.num_classes, feature_dim, feature_dim))

        features, targets = self._extract_vectors(model, train_dataloader)

        for class_idx in dataset.task_classes[task]:

            mask = targets == class_idx
            class_features = features[mask]

            class_mean = torch.mean(class_features, dim=0)
            class_cov = torch.cov(class_features.T)+torch.eye(feature_dim)*1e-4

            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov

    def _extract_vectors(self, extractor, loader):

        extractor.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for input, target, _ in loader:
                input = input.to(self.args.device)

                vector = extractor.visual.get_visual_feature(input)
                vector = vector.cpu()

                vectors.append(vector)
                targets.append(target)

        del extractor

        return torch.cat(vectors), torch.cat(targets)

    def train_phase2(self, model, dataset, task, network_params):
        self.logit_norm = self.args.logit_norm
        model.train()
        for n, p in model.named_parameters():
            if n not in ['visual.proj', 'text_projection']:
                p.requires_grad = False

        run_epochs = self.args.ca_epochs
        seen_classes_num = sum([len(dataset.task_classes[i]) for i in range(task+1)])

        optimizer = self.get_optimizer(network_params, run_epochs)

        for _ in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_per_class = 256

            for order_id, class_id in enumerate(np.concatenate(dataset.task_classes[:task+1])):
                task_id = order_id // len(dataset.task_classes[0])
                decay = (task_id + 1) / (task + 1) * 0.1
                cls_mean = torch.tensor(self._class_means[class_id], dtype=torch.float64).to(
                    self.args.device)*(0.9+decay)
                cls_cov = self._class_covs[class_id].to(self.args.device)

                class_dist = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = class_dist.sample(sample_shape=(num_sampled_per_class,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([class_id]*num_sampled_per_class)

            sampled_data = torch.cat(sampled_data, dim=0)
            sampled_label = torch.tensor(sampled_label).long()

            inputs = sampled_data
            targets = sampled_label

            shuffle_index = torch.randperm(inputs.size(0))
            inputs = inputs[shuffle_index]
            targets = targets[shuffle_index]

            full_text = torch.cat([tokenize(f"a photo of a {c}") for c in dataset.class_name_full])

            for _iter in range(seen_classes_num):
                inputs = inputs[_iter*num_sampled_per_class:(_iter+1)*num_sampled_per_class]
                targets = targets[_iter*num_sampled_per_class:(_iter+1)*num_sampled_per_class]
                texts = full_text

                images = inputs.to(self.args.device)
                texts = texts.to(self.args.device)
                targets = targets.to(self.args.device)

                image_features = images @ model.visual.proj
                text_features = model.encode_text(texts)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                logits = model.logit_scale.exp() * image_features @ text_features.t()

                per_task_norm = torch.zeros_like(logits).cuda()
                for ttask in range(task+1):
                    cur_classes = torch.tensor(dataset.task_classes[ttask])
                    temp_norm = torch.norm(logits[:, cur_classes], p=2, dim=-1, keepdim=True) + 1e-7
                    per_task_norm[:, cur_classes] = 1/temp_norm / self.logit_norm
                logits = logits * per_task_norm

                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            self.lr_scheduler.step()
