import statistics
from tqdm import tqdm
import numpy as np
import wandb

import torch
from torch.utils.data import DataLoader

from trainer.utils import accuracy, logging, resume
from dataset.imagenet import zeroshot_classifier
from clip.clip import tokenize
from dataset.imagenet import ImageNet
from metric import AverageMeter, ClassIncrementalMetric, TaskIncrementalMetric







class FrozenCLIP(object):
    def __init__(self, args):
        self.args = args

        # self.num_classes = args.num_classes
        if args.scenario == 'class_incremental':
            METRIC = ClassIncrementalMetric
        elif args.scenario in ['domain_incremental', 'task_incremental']:
            METRIC = TaskIncrementalMetric
        else:
            raise ValueError
        self.metric = METRIC(args)
        self.unseen_metric = METRIC(args)
        self.full_metric = METRIC(args)
        self.held_out_metric = AverageMeter()


    def train(self, model, dataset, task):
        pass

    def held_out_evaluation(self, model, transform):
        testset = ImageNet(self.args.imagenet_root, transform)
        metric = AverageMeter()
        zeroshot_weights = zeroshot_classifier(testset.classes, model)
        test_dataloader = DataLoader(testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        for image, label in tqdm(test_dataloader, desc=f"Evaluation for ImageNet Validation Set",total=len(test_dataloader)):
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights
            acc = accuracy(logits, label)[0]
            metric.update(acc, image.size(0))
        return metric.avg.item()
    
    def eva_task_t(self, t, testset, model, task, text_features, text_features_full):
        zero_shot_metric = AverageMeter()
        avg_metric = AverageMeter()

        test_dataloader = DataLoader(
            testset, batch_size=self.args.batch_size, num_workers=self.args.workers)
        for (image, label, _) in tqdm(test_dataloader, desc=f"Evaluation for {t}",
                                      total=len(test_dataloader)):
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if t <= task:  # update average accuracy for current batch
                logits = 100.0 * image_features @ text_features.T
                acc = accuracy(logits, label)[0]
                avg_metric.update(acc, image.size(0))

            # update zero-shot accuracy for current batch
            logits_full = 100.0 * image_features @ text_features_full.T
            acc_full = accuracy(logits_full, label)[0]
            zero_shot_metric.update(acc_full, image.size(0))

        avg = avg_metric.avg if not torch.is_tensor(
            avg_metric.avg) else avg_metric.avg.item()
        unseen_avg = zero_shot_metric.avg if not torch.is_tensor(
            zero_shot_metric.avg) else zero_shot_metric.avg.item()

        return avg, unseen_avg, len(testset)

    def evaluation(self, model, dataset, task):

        unseen_metric = self.unseen_metric
        avg_metric = self.metric

        # the classification space is all seen classes for class incremental setting
        if self.args.scenario == 'class_incremental':
            if hasattr(dataset, 'classifier'):
                text_features_full = dataset.classifier(dataset.class_name_full,model)
            else:
                text_inputs_full = torch.cat(
                    [tokenize(f"a photo of a {c}") for c in dataset.class_name_full]).cuda()
                with torch.no_grad():
                    text_features_full = model.encode_text(text_inputs_full)
                    text_features_full /= text_features_full.norm(dim=1, keepdim=True)
                
            if task < dataset.num_tasks - 1:
                unseen_class_idx = torch.Tensor(np.concatenate(dataset.task_classes[task + 1:],axis=None)).to(torch.long) 
                text_features = text_features_full.clone().detach()
                text_features[unseen_class_idx] = 0
            else :
                text_features = text_features_full.clone().detach()
        for t in range(self.args.num_tasks):
            testset = dataset.get_dataset(t, is_train=False)
            # the classificaiton space is only the current space in domain/task incremental setting
            if self.args.scenario != 'class_incremental':
                class_name = dataset.get_class_name(t)
                text_inputs_full = torch.cat(
                    [tokenize(f"a photo of a {c}") for c in class_name]).cuda()
                with torch.no_grad():
                    text_features_full = model.encode_text(text_inputs_full)
                    text_features_full /= text_features_full.norm(
                        dim=1, keepdim=True)
                text_features = text_features_full

            acc, acc_full, n = self.eva_task_t(
                t, testset, model, task, text_features, text_features_full)

            # update for current task
            self.full_metric.update(task,t,acc_full,n=n)
            self.full_metric.update_metric(task,t)
            if t <= task:
                avg_metric.update(task, t, acc, n=n)
                avg_metric.update_metric(task, t)
            else:
                unseen_metric.update(task, t, acc_full, n=n)
                unseen_metric.update_metric(task, t)
            if self.args.report_to:
                logging('task', task, f'{t}/accuracy per task', acc, self.args)

        held_out = self.held_out_evaluation(model, dataset.transform) if not (
            self.args.debug  or task != self.args.num_tasks-1 ) else 0
        self.held_out_metric.update(held_out)


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

    def only_evaluation(self, model, dataset, task):
        model, _, _, _ = resume(self.args, task, model)
        self.evaluation(model, dataset, task)

    def save_checkpoint(self, model, task, args):
        pass
