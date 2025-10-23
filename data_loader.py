#数据准备和预处理
#下载、预处理、构建DataLoader

from torchvision import datasets, transforms
from transformers import ViTImageProcessor
from torch.utils.data import DataLoader
import config

def get_dataloaders():
    '''创建并返回训练、验证、测试的数据加载器'''
    processor = ViTImageProcessor.from_pretrained(config.local_model_path)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(processor.size['height']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(processor.size['height']),
            transforms.CenterCrop(processor.size['height']),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        ])
    }

    #加载数据集
    image_datasets = {
        'train': datasets.Flowers102(root=config.DATA_DIR, split='train', download=True, transform=data_transforms['train']),
        'val': datasets.Flowers102(root=config.DATA_DIR, split='val', download=True, transform=data_transforms['val']),
        'test': datasets.Flowers102(root=config.DATA_DIR, split='test', download=True, transform=data_transforms['val']),
    }

    #创建数据加载器
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        'test': DataLoader(image_datasets['test'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names