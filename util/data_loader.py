import os
import torch
import torchvision
import numpy as np

from torchvision import transforms, datasets
from MBM.better_mistakes.data.transforms import train_transforms, val_transforms


def is_sorted(x):
    return x == sorted(x)


def train_data_loader(opts, pin_memory=True):
    if opts.data == "cifar-100":
        train_dir = opts.data_path
        transform_train = transforms.Compose([RandomPadandCrop(32), RandomFlip(), ToTensor()])
        transform_val = transforms.Compose([ToTensor()])
        train_dataset, val_dataset = get_cifar100(train_dir, transform_train=transform_train, transform_val=transform_val)
    else:
        train_dir = os.path.join(opts.data_path, "train_images")
        val_dir = os.path.join(opts.data_path, "val_images")
        train_dataset = datasets.ImageFolder(train_dir, train_transforms(opts.target_size, opts.data, augment=opts.data_augmentation, normalize=True))
        val_dataset = datasets.ImageFolder(val_dir, val_transforms(opts.data, normalize=True, resize=opts.target_size))

    assert train_dataset.classes == val_dataset.classes

    # check that classes are loaded in the right order

    assert is_sorted([d[0] for d in train_dataset.class_to_idx.items()])
    assert is_sorted([d[0] for d in val_dataset.class_to_idx.items()])
    print("batch size: ", opts.batch_size)
    # get data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers, pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers, pin_memory=pin_memory,
        drop_last=True
    )

    return train_dataset, val_dataset, train_loader, val_loader


def test_data_loader(opts, pin_memory=True):

    if opts.data == "cifar-100":
        test_dir = opts.data_path
        transform_test = transforms.Compose([ToTensor()])
        test_dataset = get_cifar100(test_dir, test=True, transform_val=transform_test)
    else:
        test_dir = os.path.join(opts.data_path, "test_images")
        test_dataset = datasets.ImageFolder(test_dir, val_transforms(opts.data, normalize=True, resize=opts.target_size))

    # check that classes are loaded in the right order
    assert is_sorted([d[0] for d in test_dataset.class_to_idx.items()])
    # get data loaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers, pin_memory=pin_memory,
        drop_last=True
    )

    return test_dataset, test_loader


def get_cifar100(root, test=False, transform_train=None, transform_val=None, download=True):
    if not test:
        base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)
        train_labeled_idxs, val_idxs = train_val_split(base_dataset.targets, len(base_dataset.classes))

        train_labeled_dataset = CIFAR100_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
        val_dataset = CIFAR100_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
        print(f"#Labeled: {len(train_labeled_idxs)} #Val: {len(val_idxs)}")
        return train_labeled_dataset, val_dataset
    else:
        test_dataset = CIFAR100_labeled(root, train=False, transform=transform_val, download=True)

        print(f"#Test: {len(test_dataset)}")
        return test_dataset


def train_val_split(labels, num_classes):
    labels = np.array(labels)
    train_labeled_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        # train val partition shall not vary with different seed
        # np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:-50])
        val_idxs.extend(idxs[-50:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(val_idxs)
    return train_labeled_idxs, val_idxs


cifar100_mean = (0.5071, 0.4867, 0.4408)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar100_std = (0.2675, 0.2565, 0.2761)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar100_mean, std=cifar100_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """

    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """

    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class CIFAR100_labeled(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
