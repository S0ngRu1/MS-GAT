import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from model.image_encoder import ImageEncoder
from utils.process_data import get_transforms

class ImageDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.save = []
        self.image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder).get_tokenizer()
        self.img_width = 224
        self.img_height = 224
        self.depth = 3
        self.transforms = get_transforms()
        if args.dataset in ['Weibo17','Weibo21']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '.csv', encoding='utf-8').dropna()
        elif args.dataset in ['CFND_dataset']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '_data.csv', encoding='utf-8').dropna()
        else:
            logger.error(f'无效数据集: {args.dataset}，支持的数据集为 ["Weibo17", "Weibo21", "CFND_dataset"]')
            raise ValueError("无效数据集，请检查参数")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = ''
        label = None
        if self.args.dataset in ['Weibo17','Weibo21']:
            image_name, label = self.df.iloc[index,[1,3]].values
            img_path = os.path.join(self.args.data_dir, self.args.dataset, 'new_images', image_name)
        elif self.args.dataset in ['CFND_dataset']:
            image, label = self.df.iloc[index,[2,3]].values
            img_path = os.path.join(self.args.data_dir,self.args.dataset, image)
        else:
            logger.error(f'无效数据集: {self.args.dataset}，支持的数据集为 ["Weibo17", "Weibo21", "CFND_dataset"]')
            raise ValueError("无效数据集，请检查参数")
        try:
            if os.path.exists(img_path) and os.path.isfile(img_path):
                image = Image.open(img_path).convert("RGB")
                image = self.transforms(image)
                img_inputs = self.image_tokenizer(images=image, return_tensors="pt").pixel_values
                return img_inputs, label
            else:
                print(f"Image {img_path} does not exist. Skipping.")
                return None 
        except OSError as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
            return None  
        
    

def custom_collate_fn_imagedata(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return [], []
    img_inputs = torch.cat([item[0] for item in valid_batch], dim=0)  
    labels = torch.tensor([item[1] for item in valid_batch])  
    return img_inputs, labels
        
def image_data_loader(args):
    if args.dataset in ['Weibo17','Weibo21','CFND_dataset']:
        train_set = ImageDataset(args, mode='train')
        valid_set = ImageDataset(args, mode='val')
        test_set = ImageDataset(args, mode='test')
        logger.info(f'Train Dataset: {len(train_set)}')
        logger.info(f'Valid Dataset: {len(valid_set)}')
        logger.info(f'Test Dataset: {len(test_set)}')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True, collate_fn=custom_collate_fn_imagedata)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True, collate_fn=custom_collate_fn_imagedata)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True, collate_fn=custom_collate_fn_imagedata)
        return train_loader, valid_loader, test_loader
    else:
        logger.info('数据集无效')
        return None, None, None
