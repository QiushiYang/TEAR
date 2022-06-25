import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
from datasets import transform as T
from datasets.randaugment import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def get_dataLoader_mix(transformSequence=None, trans_aug=None, trans_strong=None, labelled=50, mu=1, batch_size=8,
                       txtFilePath='./datasets/Labels',
                       pathDirData =  '/home/qsyang2/data/ISIC2018/ISIC2018_Task3_Training_Input/'):

    pathFileTrain_L =  txtFilePath + '/Train' + str(labelled) + '.txt'
    pathFileTrain_U =  txtFilePath + '/Train_unl' + str(labelled) + '.txt'
    validation =  txtFilePath + '/Val.txt'
    test =  txtFilePath + '/Test.txt'

    mean = (0.21429618, 0.21459657, 0.21503997)
    std = (0.3269253, 0.32702848, 0.32728335)

    trans_aug = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.RandomRotation(degrees=(-10,10)),
            transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    
    trans_strong = T.Compose([
                    T.ToRGB(),
                    T.Resize((128, 128)),  ###
                    T.PadandRandomCrop(border=4, cropsize=(128, 128)),
                    T.RandomHorizontalFlip(p=0.5),
                    RandomAugment(2, 10), 
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])
    
    transformSequence = T.Compose([
                T.ToRGB(),
                T.Resize((128, 128)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])


    datasetTrainLabelled = DatasetGenerator_Mix(path=pathDirData, textFile=[pathFileTrain_L],
                                                    transform=trans_aug)
    datasetTrainUnLabelled = DatasetGenerator_Mix(path=pathDirData, textFile=[pathFileTrain_U],
                                                    transform=trans_aug, strong_transform=trans_strong)
    datasetVal = DatasetGenerator_Mix(path=pathDirData, textFile=[validation],
                                                    transform=transformSequence)
    datasetTest = DatasetGenerator_Mix(path=pathDirData, textFile=[test],
                                                    transform=transformSequence)


    dataLoaderTrainLabelled = DataLoader(dataset=datasetTrainLabelled, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    dataLoaderTrainUnLabelled = DataLoader(dataset=datasetTrainUnLabelled, batch_size=batch_size*mu, shuffle=True, num_workers=1, drop_last=True)
    dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return dataLoaderTrainLabelled, dataLoaderTrainUnLabelled, dataLoaderVal, dataLoaderTest

class DatasetGenerator_Mix(Dataset):
    def __init__(self, path, textFile, transform, strong_transform=None):
        self.listImagePaths = []
        self.listImageLabels = []
        self.weak_transform = transform
        self.strong_transform = strong_transform

        pathDatasetFile = textFile[0]
        fileDescriptor = open(pathDatasetFile, "r")
        line = True

        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                nameSplit = lineItems[0]
                if 'test' in path:
                    imagePath = os.path.join(path, nameSplit)
                else:
                    imagePath = os.path.join(path, nameSplit)
                imageLabel = lineItems[1:]
                imageLabel = [int(float(i)) for i in imageLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)
        fileDescriptor.close()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath)
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        
        if self.strong_transform != None: 
            weak_imageData = self.weak_transform(imageData)
            strong_imageData = self.strong_transform(imageData) 
            return [weak_imageData, strong_imageData], imageLabel
        else:
            imageData = self.weak_transform(imageData)
            return imageData, imageLabel

    def __len__(self):
        return len(self.listImagePaths)


if __name__ == '__main__':
    #Transforms for the data
    import torchvision.transforms as transforms
    #Transforms for the data
    mean = (0.21429618, 0.21459657, 0.21503997)
    std = (0.3269253, 0.32702848, 0.32728335)
    
    trans_aug = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.RandomRotation(degrees=(-10,10)),
            transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
            transforms.ToTensor()
            ])
    
    trans_strong = T.Compose([
                    T.Resize((128, 128)),  ###
                    T.PadandRandomCrop(border=4, cropsize=(128, 128)),
                    T.RandomHorizontalFlip(p=0.5),
                    RandomAugment(2, 10),
                    T.ToTensor(),
                ])
    
    transformSequence = T.Compose([
                T.Resize((128, 128)),
                T.ToTensor(),
            ])
    
    labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = get_dataLoader_mix(
        transformSequence, trans_aug, trans_strong, labelled=50, mu=1, batch_size=1)

    for ii, (sample, label) in enumerate(labeled_trainloader):
        for jj in range(sample.size()[0]):
            print(sample[jj].shape)
            print(label[jj])
        if ii == 1000:
            break
