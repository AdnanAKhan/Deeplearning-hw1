import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

NET = 'resnet18'
SIZE = {
    'resnet18': 224,
}

train_transformer = transforms.Compose([
    transforms.Resize(size=(SIZE[NET], SIZE[NET])),  # resize the image to appropriate shape
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  # transforms it into a torch tensor

eval_transformer = transforms.Compose([
    transforms.Resize(size=(SIZE[NET], SIZE[NET])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  # transforms it into a torch tensor

test_transformer = transforms.Compose([
    transforms.Resize(size=(SIZE[NET], SIZE[NET])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  # transforms it into a torch tensor


class LocalizationDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, data_dir, transform=None, mode='train'):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """

        # self.raw_image_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd())),
        #                                 'raw_dataset', 'Localization dataset', 'images')

        self.raw_image_dir = '/import/helium-share/staff/ywang/comp7950/images'
        self.mode = mode

        self.dataset_df = pd.read_csv(os.path.join(data_dir, 'dataset.csv'), header=0)

        self.filenames = []
        self.labels = []

        for ind, row in self.dataset_df.iterrows():
            self.filenames.append(row['filename'])

            if self.mode == 'test':
                pass
            else:
                self.labels.append(np.array([row['x'], row['y'], row['w'], row['h']], dtype=np.float))

        if transform:
            self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(os.path.join(self.raw_image_dir, self.filenames[idx])).convert('RGB')  # PIL image
        width, height = image.size

        if self.transform:
            image = self.transform(image)

        if self.mode == 'test':
            return image, np.array((0, 1), np.float), np.array([width, height], dtype=np.float)
        else:
            return image, self.labels[idx], np.array([width, height], dtype=np.float)


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyper parameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            dataset_csv_path = os.path.join(data_dir, "{}".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(LocalizationDataset(dataset_csv_path, train_transformer, mode='train'),
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
            elif split == 'val':
                dl = DataLoader(LocalizationDataset(dataset_csv_path, eval_transformer, mode='val'),
                                batch_size=params.batch_size,
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            elif split == 'test':
                dl = DataLoader(LocalizationDataset(dataset_csv_path, test_transformer, mode='test'),
                                batch_size=1,
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            else:
                # not handling the test in this case.
                dl = None

            if dl:
                dataloaders[split] = dl

    return dataloaders
