
import numpy as np
from torch.utils.data import Dataset
import datasets

from dataset.cifar100 import SplitCifar100
from clip.clip import tokenize


order = [51,  94,  55,  35, 170, 177,  20,  85,  50,  36,  30,  76,   5, 136, 182,  82,  25, 169, 166, 178,
         74,  53,  32, 184, 160, 179, 138, 140,  27,  12,  48,  57, 145,  28,  19, 162, 175, 121,  18,  72,
         101,  69,  49, 115, 181,  15, 193,  37, 111,   0, 158,  33,  11,  47,  80, 126, 183,  16, 198,  91,
         58,  70,   2,  67,   8, 199,  10,   3,  77,  22, 168,  96,  86,   4, 189,  88,  99,  31,  84,  17,
         107, 123,  29, 103, 117, 161, 105,  73, 173,  13,  24,   1, 195, 185,  79,  87, 151,  65,  62,  26,
         147, 144,  52,  75, 186, 159, 109,  66, 137, 191, 122, 133, 142,  38,  39,  61,  98, 157, 192, 129,
         112, 197, 149, 194, 104, 152, 120,  56, 124, 132,  89, 141, 116, 146, 153, 176, 127,  71, 125,  63,
         135, 118, 102,  41, 150, 154,  90, 172, 167, 106, 114,  46, 165, 131, 196, 156, 180,  34,  44,  83,
         164,   6,  59,  60,  45, 143,  42, 134, 108,  97,  81, 119,  93,   7, 187,  68, 128, 113, 139,  95,
         130, 100, 163, 110,  40, 174, 148,   9, 190,  54, 155,  64,  78, 171, 188,  43,  92,  21,  23,  14]


class CLIPDataset(Dataset):
    _repr_indent = 4

    def __init__(self, set, text, idx, transform, **kwargs):
        self.data = set
        self.text = text
        self.idx = idx
        self.transform = transform
        self.classes = np.array([self.data.targets[i] for i in idx])
        self.data.transform = transform
        self.cub_name = []
        with open('./dataset/cub_name.txt', 'r') as f:
            for line in f.readlines():
                name = line.strip().split('.')[-1].replace("_", ' ')
                self.cub_name.append(name)

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
        name = self.cub_name[label]
        text = tokenize(f'a photo of a {name}')[0]
        return image, label, text





class SplitCUB(SplitCifar100):
    def __init__(self, args, root, transform=None, num_tasks=None):

        self.trainset = datasets.load_dataset('alkzar90/CC6204-Hackaton-Cub-Dataset', cache_dir=root, split='train')
        self.testset = datasets.load_dataset('alkzar90/CC6204-Hackaton-Cub-Dataset', cache_dir=root, split='test')
        self.transform = transform
        self.root = root

        self.task = 0
        self.mode = 'train'
        self.classes = []  # seen class names
        self.buffer = {}

        self.num_classes = 200
        self.num_tasks = args.num_tasks if num_tasks is None else num_tasks
        if args.joint:
            self.num_tasks = 1
        self.buffer_size = int(240 * args.buffer_size)
        self.scenario = 'class_incremental'

        self.task = 0
        self.mode = 'train'
        self.set = self.trainset

        self.paths = {}

        self.class_name_full = []
        with open('./dataset/cub_name.txt', 'r') as f:
            for line in f.readlines():
                name = line.strip().split('.')[-1].replace("_", ' ')
                self.class_name_full.append(name)
        print(self.class_name_full)


        self.trainset.targets = self.trainset['label']
        self.testset.targets = self.testset['label']

        classes = order

        self.task_classes = np.array_split(classes, self.num_tasks)
        print('task split', self.task_classes)


        self.dataset_collect_fcn = CLIPDataset


        self._get_image_list_for_cur_set()

