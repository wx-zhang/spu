
import os
import json
from PIL import Image

from torch.utils.data import Dataset

from clip.clip import tokenize


def default_loader(path: str):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# some images could be corrupted or missing during the download process
# load your valid list of images here
# you can save your valid list by the save_valid()


class cc12m(Dataset):
    def __init__(self, transform=None, root=None):
       
        
        
        valid_file = './valid.json'
        
        with open(valid_file, 'r') as f:
            valid_list = json.load(f)

        self.root = root
        self.meta = valid_list
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        name = self.meta[index].split('---')[0]
        if len(name) < 9:
            name = '{:0>9}'.format(name)
        img_path = os.path.join(self.root, name + '.jpg')
        image = default_loader(img_path)
        if self.transform:
            image = self.transform(image)

        text_path = os.path.join(self.root, name + '.txt')
        with open(text_path, 'r') as f:
            txt = f.readline()
        return image, -1, tokenize(txt,truncate=True)[0]


def save_valid():
    import datasets
    cc12m_meta = datasets.load_dataset('flax-community/conceptual-12m-multilingual-marian-es',
                                       split=datasets.Split.VALIDATION)
    valid_list = []
    for i in range(len(cc12m_meta)):
        name = cc12m_meta[i]['image_file'].split('---')[0]
        if len(name) < 9:
            name = '{:0>9}'.format(name)
        root = '/path/to/your/cc12m/root'
        img_path = os.path.join(root, name + '.jpg')
        if os.path.isfile(img_path):
            valid_list.append(name)

    import json

    with open('valid.json', 'w') as f:
        json.dump(valid_list, f)
