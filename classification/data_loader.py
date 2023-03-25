import os

from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import ImageFolder

def load_dataset(
        data_transforms,
        dataset_dir='./dataset',
        dataset_name='catdog',
        is_train=True,
):
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    is_train_dir = 'train' if is_train else 'test'

    dataset = ImageFolder(
        os.path.join(dataset_dir, is_train_dir),
        data_transforms[is_train_dir],
    )
    return dataset


def get_loaders(config, input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
    }

    dataset_name = config.dataset_name
    train_set = load_dataset(
        data_transforms, dataset_name=dataset_name, is_train=True
    )
    test_set = load_dataset(
        data_transforms, dataset_name=dataset_name, is_train=True
    )

    # Shuffle dataset to split into valid/test set.
    train_set, valid_set, test_set = divide_dataset(
        train_set, test_set, config.dataset_name,
        config.train_ratio, config.valid_ratio, config.test_ratio
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True, # training은 무조건 shuffling, 안하면 학습이 되지 않음
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=False, # 보통 False, 일부 수정된 부분에 대한 동작/결과를 보고자 할 때,
                       # 같은 테스트셋으로 진행해야 그 수정이 올바르게 되었는지 알 수 있음
    )

    return train_loader, valid_loader, test_loader


def divide_dataset(
        train_set,
        test_set,
        data_name='catdog',
        train_ratio=.6,
        valid_ratio=.2,
        test_ratio=.2,
):
    if data_name == 'catdog': # test set은 labeling이 안되어 있기 때문에, train set으로 새로 만듦
        train_cnt = int(len(train_set) * train_ratio)
        valid_cnt = int(len(train_set) * valid_ratio)
        test_cnt = len(train_set) - train_cnt - valid_cnt

        train_set, valid_set, test_set = random_split(
            train_set,
            [train_cnt, valid_cnt, test_cnt]
        )

    elif data_name == 'hymenoptera':
        valid_ratio = valid_ratio / (train_ratio + valid_ratio)
        valid_cnt = int(len(train_set) * valid_ratio)
        train_cnt = len(train_set) - valid_cnt

        train_set, valid_set = random_split( #dataset 객체를 random하게 split하는 함수
            train_set,
            [train_cnt, valid_cnt]
        )
    else:
        raise NotImplementedError('You need to specify dataset name.')

    return train_set, valid_set, test_set