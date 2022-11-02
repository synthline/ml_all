from torch import optim
from trainers import PyTorchTrainer
from networks import *
from torchvision import transforms

def train_pytorch_MLPBasic():

    transform = transforms.ToTensor() ## simply a conversion from PIL (image format) to torch tensors.
    ## down below is an example of a composition of different ways to augment the training data.
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomAffine(45),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor()
    # ])
    network = MLPBasic()

    trainerMLPBasic = PyTorchTrainer(
        nn_module=network,
        transform=transform,
        optimizer=optim.SGD(network.parameters(), lr=1e-2, momentum=0.5),
        batch_size=128,
    ) 

    print("training MLP started")
    trainerMLPBasic.train(10)
    trainerMLPBasic.save()
    print("training MLP ended")


train_pytorch_MLPBasic()
