from torchvision import models
from torch import nn

def set_parameter_required_grad(model, freeze):
    for param in model.parameters():
        param.requires_grad = not freeze # True - 학습이 됨 / False - 학습이 안됨


def get_model(config):
    # You can also refer:
    # - https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # - https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    model = None
    input_size = 0

    if config.model_name == "resnet":
        """ Resnet34"""
        model = models.resnet34(pretrained=config.use_pretrained)
        set_parameter_required_grad(model, config.freeze) # 모델 전체의 weights를 freeze

        # n_features = model.fc.in_features: model의 fc 레이어의 in_features 속성을 사용하여 이전 레이어의 출력 크기를 가져옵니다.
        # model.fc = nn.Linear(n_features, config.n_classes): model의 fc 레이어를 nn.Linear 객체로 교체합니다.
                                                            # n_features를 입력 크기로 사용하고, config.n_classes를 출력 크기로 사용합니다.
                                                            # 이를 통해 model은 입력 크기를 고정하면서 출력 크기를 변경할 수 있습니다.
        n_features = model.fc.in_features # 이전 layer
        model.fc = nn.Linear(n_features, config.n_classes) # 마지막 layer (softmax): 변경 대상
        input_size = 224 # 224를 가정한 architecture로 다른 숫자 넣으면 error
    elif config.model_name == "alexnet":
        """Alexnet"""
        model = models.alexnet(pretrained=config.use_pretrained)
        set_parameter_required_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)
        input_size = 224
    elif config.model_name == "vgg":
        """VGG16_bn"""
        model = models.vgg16_bn(pretrained=config.use_pretained)
        set_parameter_required_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)
        input_size = 224
    elif config.model_name == "squeezenet":
        """Squeezenet"""
        model = models.squeezenet1_0(pretrained=config.use_pretrained)
        set_parameter_required_grad(model, config.freeze)

        model.classifier[1] = nn.Conv2d(
            512,
            config.n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

        model.n_classes = config.n_classes
        input_size = 224
    elif config.model_name == "densenet":
        """Densenet"""
        model = models.densenet121(pretraiend=config.use_pretrained)
        set_parameter_required_grad(model, config.freeze)

        n_features = model.classifier.in_features
        model.classifier = nn.Linear(n_features, config.n_classes)
        input_size = 224
    else:
        raise NotImplementedError('You need to specify model name')

    return model, input_size



