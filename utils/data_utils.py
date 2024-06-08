from utils.datasets import CIFAR10, CIFAR100, CIFAR100_openset, CIFAR10_openset, \
    MNIST_openset, SVHN_openset, TinyImageNet_OOD_nonoverlap, ImageNetR, VISDA
import json
import torch
import os
import torchvision.transforms as transforms
import numpy as np
import utils.augmix_ops as augmentations
from utils.robustbench_loader import CustomImageFolder




def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = []
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            classname = " ".join(line[1:])
            classnames.append(classname)
    return classnames


def get_weak_ood_data(args, te_transforms, tesize=10000):
    """Return data dict(containing class names; no. of classes) and dataset.
    """
    data_dict = {}
    if args.weak_OOD == 'cifar10OOD':
        data_dict['ID_classes'] = list(json.load(open(f'datasets/cifar10_prompts_full.json')).keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        print('Test on %s level %d' %(args.corruption, args.level))
        teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
        teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
        weak_ood_dataset = CIFAR10(root=args.dataroot,
                        train=False, download=True, transform=te_transforms)
        weak_ood_dataset.data = teset_raw_10

    elif args.weak_OOD == 'cifar100OOD':
        data_dict['ID_classes'] = list(json.load(open(f'datasets/cifar100_prompts_full.json')).keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        print('Test on %s level %d' %(args.corruption, args.level))
        teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
        teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
        weak_ood_dataset = CIFAR100(root=args.dataroot,
                        train=False, download=True, transform=te_transforms)
        weak_ood_dataset.data = teset_raw_100
        
    elif args.weak_OOD == 'ImagenetROOD':

        testset = ImageNetR(root= args.dataroot)
        data_dict['ID_classes'] = testset.classnames
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        weak_ood_dataset = ImageNetR(root= args.dataroot, transform=te_transforms, train=True, tesize=30000)

    elif args.weak_OOD == "VisdaOOD":
        data_dict['ID_classes'] = json.load(open(f'datasets/visda_classes.json'))['classnames']
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        weak_ood_dataset = VISDA(root= f'{args.dataroot}/visda-2017', label_files=f'datasets/visda_validation_list.txt' , transform=te_transforms, tesize=50000)
        
    elif args.weak_OOD == 'ImagenetCOOD':
        imagenet_classes = read_classnames(f'datasets/imagenet_classnames.txt')
        data_dict['ID_classes'] = imagenet_classes
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        corruption_dir_path = os.path.join(args.dataroot, 'ImageNet-C', args.corruption,  str(args.level))
        weak_ood_dataset = CustomImageFolder(corruption_dir_path, te_transforms, tesize=tesize)

    return data_dict, weak_ood_dataset



def get_strong_ood_data(args, te_transforms, tesize=10000):

    if args.strong_OOD == 'MNIST':
        te_rize = transforms.Compose([transforms.Grayscale(3), te_transforms ])
        strong_ood_dataset = MNIST_openset(root=args.dataroot,
                    train=True, download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)
    
    elif args.strong_OOD =='SVHN': 
        te_rize = transforms.Compose([te_transforms ])
        strong_ood_dataset = SVHN_openset(root=args.dataroot,
                    split='train', download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)

    elif args.strong_OOD =='cifar10':
        teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
        teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
        strong_ood_dataset = CIFAR10_openset(root=args.dataroot,
                        train=True, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
        strong_ood_dataset.data = teset_raw_10[:int(tesize*args.strong_ratio)]

    elif args.strong_OOD =='cifar100':
        teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
        teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
        strong_ood_dataset = CIFAR100_openset(root=args.dataroot,
                        train=True, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
        strong_ood_dataset.data = teset_raw_100[:int(tesize*args.strong_ratio)]

    elif args.strong_OOD =='Tiny':

        transform_test = transforms.Compose([te_transforms ])
        strong_ood_dataset = TinyImageNet_OOD_nonoverlap(args.dataroot +'/tiny-imagenet-200', transform=transform_test, train=True)

    return strong_ood_dataset


def prepare_ood_test_data(args, te_transforms):

    data_dict, weak_ood_dataset = get_weak_ood_data(args, te_transforms, tesize=args.tesize)
    strong_ood_dataset = get_strong_ood_data(args, te_transforms, tesize=args.tesize)
    id_ood_dataset = torch.utils.data.ConcatDataset([weak_ood_dataset, strong_ood_dataset])
    print(f'weak ood data: {len(weak_ood_dataset)};  strong ood data: {len(strong_ood_dataset)}')
    
    ID_OOD_loader = torch.utils.data.DataLoader(id_ood_dataset, batch_size=args.batch_size, shuffle=True)

    return data_dict, id_ood_dataset, ID_OOD_loader


# TPT Transforms

# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()   # Resizing with scaling and ratio
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views
