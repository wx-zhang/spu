import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.cars_name import classes as car_names
from dataset.cars_name import templates,order
from dataset.cifar100 import SplitCifar100

sys.path.append("..")
import datasets

from clip.clip import tokenize


class CLIPDataset(Dataset):
    _repr_indent = 4

    def __init__(self, set, text, idx, transform, **kwargs):
        self.data = set
        self.text = text
        self.idx = idx
        self.transform = transform
        self.classes = np.array([self.data.targets[i] for i in idx])
        self.data.transform = transform
    def __len__(self):
        return len(self.idx)

    def __repr__(self) -> str:
        head = "Dataset "
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += [f"Current seen classes {len(np.unique(self.classes))}"]
        if hasattr(self.data, "transform") and self.data.transform is not None:
            body += [repr(self.data.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def __getitem__(self, idx):
        index = int(self.idx[idx])
        image = self.transform(self.data[index]['image'])
        label = int(self.data[index]['label'])
        name = car_names[label]
        text = tokenize(f'a photo of a {name}')[0]
        return image, label, text
    


class SplitCars(SplitCifar100):
    def __init__(self, args, root, transform=None,num_tasks=None):
        self.trainset = datasets.load_dataset('Multimodal-Fatima/StanfordCars_train', cache_dir=root)['train']
        self.testset = datasets.load_dataset('Multimodal-Fatima/StanfordCars_test', cache_dir=root)['test']
        self.transform = transform
        self.root = root

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}

        self.num_classes = 196
        self.num_tasks =  args.num_tasks if num_tasks is None else num_tasks
        if args.joint:
            self.num_tasks = 1
        self.buffer_size = int(240 * args.buffer_size )
        self.scenario = 'class_incremental'

        self.task = 0
        self.mode = 'train'
        self.set = self.trainset

        self.paths = {}

  
        self.trainset.targets = self.trainset['label']
        self.testset.targets = self.testset['label']

        classes = order

        self.task_classes = np.array_split(classes, self.num_tasks)
        print('task split', self.task_classes)



        self.dataset_collect_fcn = CLIPDataset


        self._get_image_list_for_cur_set()


        self.class_name_full = car_names


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