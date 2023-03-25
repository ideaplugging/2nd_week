import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from classification.data_loader import get_loaders
from classification.trainer import Trainer
from classification.model_loader import get_model


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.backends.mps.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.6)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--test_ratio', type=float, default=.2)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--model_name', type=str, default='resnet')
    p.add_argument('--dataset_name', type=str, default='catdog')
    p.add_argument('--n_classes', type=int, default=2)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--use_pretrained', action='store_true')
    # action='store_true'는 인자가 지정되면 해당 인자가 True로 설정되도록 하는 것을 의미
    # 따라서 --freeze 또는 --use_pretrained가 지정되면 해당 인자는 True로 설정

    # train from scratch freeze와 use_pretrained를 인자로 넣지 않은 경우
    # train from pretrained weights use_pretrained만 인자로 넣은 경우
    # train with freezed pretrained weights freeze와 use_pretrained를 모두 인자로 넣은 경우

    # freeze만 쓰면 안됨 - 나중에 예외처리 해보기

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    if config.verbose >= 2:
        print(config)

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('mps:%d' % config.gpu_id)

    model, input_size = get_model(config)
    model = model.to(device)

    train_loader, valid_loader, test_loader = get_loaders(config, input_size)

    # 몇개의 데이터가 있는지 출력
    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)


if __name__ == '__main__':
    config = define_argparser()
    main(config)