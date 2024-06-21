
import numpy as np
import torch
from torchvision.datasets import GTSRB
from tqdm import tqdm

from clip.clip import tokenize
from dataset.cifar100 import CLIPDataset, SplitCifar100
from dataset.gtsrb_name import classes as class_names
from dataset.gtsrb_name import templates


class SplitGTSRB(SplitCifar100):
    def __init__(self, args, root, transform=None):

        self.trainset = GTSRB(root,split='train', transform=transform,download=True)
        self.testset = GTSRB(root,split='test', transform=transform,download=True)
        self.trainset.targets = [i[1] for i in self.trainset._samples]
        self.testset.targets = [i[1] for i in self.testset._samples]

        self.transform = transform
        self.root = root

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}

        self.num_classes = len(class_names)
        self.num_tasks = args.num_tasks
        if args.joint:
            self.num_tasks = 1
        self.buffer_size = int(1000 * args.buffer_size )
        self.scenario = 'class_incremental'

        self.task = 0
        self.mode = 'train'
        self.set = self.trainset

        self.paths = {}
        self.class_name_full = class_names
        classes = list(range(self.num_classes))
        classes = [25, 2, 11, 1, 40, 27, 5, 9, 17, 32, 29, 20, 39, 21, 15, 23, 10, 3, 18, 38, 42, 14, 22, 35, 34, 19, 33, 12, 26, 41, 0, 37, 6, 13, 24, 30, 28, 31, 7, 16, 4, 36, 8]


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
