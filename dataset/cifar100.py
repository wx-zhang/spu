
import copy
import random
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100


from clip.clip import tokenize
from dataset.cifar100_name import classes as class_names
from dataset.cifar100_name import templates, order


class CLIPDataset(Dataset):
    _repr_indent = 4

    def __init__(self, set, text, idx, **kwargs):
        self.data = set
        self.text = text
        self.idx = idx
        self.classes = np.array([self.data.targets[i] for i in idx])

    def __len__(self):
        return len(self.idx)

    def __repr__(self) -> str:

        head = "Dataset "
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += [f"Current seen classes {len(np.unique(self.classes))}"]
        body.append("Image root location: {}".format(self.data.root))
        if hasattr(self.data, "transform") and self.data.transform is not None:
            body += [repr(self.data.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def __getitem__(self, idx):
        index = self.idx[idx]
        image, label = self.data[index]
        name = self.text[label]
        name = name.replace('_', ' ')
        text = tokenize(f'a photo of {name}')[0]

        return image, label, text






class SplitCifar100(object):
    def __init__(self, args, root, transform=None, valid=False, num_tasks=None):
        self.trainset = CIFAR100(root, train=True, transform=transform, download=True)
        self.testset = CIFAR100(root, train=False, transform=transform, download=True)
        self.transform = transform

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}
        self.idx_to_class = {}

        self.valid = valid

        self.num_classes = 100
        self.num_tasks = args.num_tasks if num_tasks is None else num_tasks
        if args.joint:
            self.num_tasks = 1
        self.buffer_size = int(2000 * args.buffer_size)
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
        self.classifier = zeroshot_classifier

        self._get_image_list_for_cur_set()

    def get_dataset(self, task_id, is_train=True, with_buffer=True, balanced=False):
        if balanced:
            with_buffer = False
        self.task = task_id
        if is_train:
            self.mode = 'train'
            self.set = self.trainset
        else:
            self.mode = 'test'
            self.set = self.testset
        self._get_image_list_for_cur_set(with_buffer=with_buffer)
        idx = copy.deepcopy(self.data_idx)
        curset = self.dataset_collect_fcn(self.set, self.class_name_full, idx, transform=self.transform)

        return curset

    def get_buffer(self, task):
        self.set = self.trainset
        assert (task-1 in self.buffer.keys())
        self.data_idx = []
        for key in range(task):
            self.data_idx.extend(self.buffer[key]['data_idx'])
        idx = copy.deepcopy(self.data_idx)
        return self.dataset_collect_fcn(self.set, self.class_name_full, idx, transform=self.transform)

    def get_task_data_from_buffer(self, task):
        self.set = self.trainset
        assert (task in self.buffer.keys())
        self.data_idx = []
        self.data_idx.extend(self.buffer[task]['data_idx'])
        idx = copy.deepcopy(self.data_idx)
        return self.dataset_collect_fcn(self.set, self.class_name_full, idx, transform=self.transform)

    def _get_image_list_for_cur_set(self, with_buffer=True):
        if f'{self.task}_{self.mode}' in self.paths.keys():
            # if we have already read paths and labels from fils
            self.data_idx = self.paths[f'{self.task}_{self.mode}']['data_idx']
        else:
            targets = self.set.targets
            # prepare data idx for current set
            self.data_idx = []
            for idx in range(len(targets)):
                if targets[idx] in self.task_classes[self.task]:
                    self.data_idx.append(idx)

            # and save to self.path
            self.paths[f'{self.task}_{self.mode}'] = {}
            self.paths[f'{self.task}_{self.mode}']['data_idx'] = self.data_idx

        if (self.buffer is not {}) and (self.mode == 'train') and (with_buffer):
            for key in self.buffer.keys():
                self.data_idx.extend(self.buffer[key]['data_idx'])

    def update_buffer(self, task):
        '''
        update the buffer with data from task,
        the buffer is task balanced such that
        it always contains the same number of
        samples from different tasks.
        :param task: update buffer with samples from which task,

        '''
        # make sure we have already read path file for the task and did not update buffer with this task
        assert (f'{task}_train' in self.paths.keys()) and (task not in self.buffer.keys())
        cur_buffer_task_length = self.buffer_size // (len(self.buffer.keys()) + 1)

        # cut buffer sample from previous task
        for key in self.buffer.keys():
            pre_task_length = len(self.buffer[key]['data_idx'])
            cur_task_length = min(cur_buffer_task_length, pre_task_length)
            indices = random.sample(range(pre_task_length), cur_task_length)
            self.buffer[key]['data_idx'] = [self.buffer[key]['data_idx'][i] for i in indices]
        # update buffer with current task
        task_length = len(self.paths[f'{task}_train']['data_idx'])
        cur_task_length = min(cur_buffer_task_length, task_length)
        indices = random.sample(range(task_length), cur_task_length)

        key = task
        self.buffer[key] = {}
        self.buffer[key]['data_idx'] = [self.paths[f'{task}_train']['data_idx'][i] for i in indices]
        self.current_buffer_length = sum([len(self.buffer[key]['data_idx']) for key in self.buffer.keys()])


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
