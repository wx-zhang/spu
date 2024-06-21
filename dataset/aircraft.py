from clip.clip import tokenize
import sys

import torch
from tqdm import tqdm
import numpy as np
from torchvision.datasets import FGVCAircraft
from dataset.aircraft_name import classes as class_names
from dataset.aircraft_name import templates, order
from dataset.cifar100 import SplitCifar100, CLIPDataset

sys.path.append("..")


class SplitAircraft(SplitCifar100):
    def __init__(self, args, root, transform=None):
        self.trainset = FGVCAircraft(root, split='trainval', transform=transform)
        self.testset = FGVCAircraft(root, split='test', transform=transform)
        self.trainset.targets = self.trainset._labels
        self.testset.targets = self.testset._labels
        self.transform = transform
        self.root = root

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}

        self.num_classes = 100
        self.num_tasks = args.num_tasks
        if args.joint:
            self.num_tasks = 1
        self.buffer_size = int(250 * args.buffer_size)
        self.scenario = 'class_incremental'

        self.task = 0
        self.mode = 'train'
        self.set = self.trainset

        self.paths = {}
        self.class_to_idx = self.trainset.class_to_idx
        self.class_name_full = class_names
        classes = order

        self.task_classes = np.array_split(classes, self.num_tasks)
        print('task split', self.task_classes)


        self.dataset_collect_fcn = CLIPDataset


        self._get_image_list_for_cur_set()

        self.classifier = zeroshot_classifier


def zeroshot_classifier(classnames, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates]  # format with class
            texts = tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(
                texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.T
