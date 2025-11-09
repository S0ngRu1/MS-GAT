import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from loguru import logger
from model.text_encoder import TextEncoder
from model.image_encoder import ImageEncoder
from model.clip_encoder import ClipEncoder
from model.fft import extract_spectral_features
from model.sentiment_analysis import SentimentConfig , SentimentModel, predict_one as sentiment_predict
from utils.process_data import get_transforms, preprocess_text


class MMDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.save = []
        self.max_length = 512
        if args.dataset in ['Weibo17','Weibo21']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '.csv', encoding='utf-8').dropna()
        elif args.dataset in ['CFND_dataset']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '_data.csv', encoding='utf-8').dropna()
        else:
            logger.error(f'无效数据集: {args.dataset}，支持的数据集为 ["Weibo17", "Weibo21", "CFND_dataset"]')
            raise ValueError("无效数据集，请检查参数")
        if self.args.method in ['FND-2-CLIP']:
            self.clip_encoder = ClipEncoder(self.args, pretrained_dir=args.pretrained_dir)
        else:
            self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder).get_tokenizer()
            self.image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder).get_tokenizer()
        self.img_width = 224
        self.img_height = 224
        self.depth = 3
        self.transforms = get_transforms()
        self.sentiment_config = SentimentConfig()
        self.sentiment_config.device = args.device
        self.sentiment_model = SentimentModel(self.sentiment_config).to(self.sentiment_config.device)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = None
        label = None
        img_path = ''
        if self.args.dataset in ['Weibo17','Weibo21']:
            image_name, text, label = self.df.iloc[index, 1:4].values
            img_path = os.path.join(self.args.data_dir, self.args.dataset, 'new_images', image_name)
        
        elif self.args.dataset in ['CFND_dataset']:
            text, image, label = self.df.iloc[index, 1:4].values
            img_path = os.path.join(self.args.data_dir,self.args.dataset, image)
    
        text = preprocess_text(text)
        sentiment_output = sentiment_predict(text, self.sentiment_config, self.sentiment_model)
        if self.args.method not in ['FND-2-CLIP', 'FND-2-SGAT']:
            text_tokens = self.text_tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True,
                                        padding='max_length', return_tensors="pt")
            try:
                if os.path.exists(img_path) and os.path.isfile(img_path):
                    image = Image.open(os.path.join(img_path)).convert("RGB")
                    image = self.transforms(image)
                    fft_image = extract_spectral_features(img_path,output_dim=(128, 128))
                    fft_image = torch.tensor(fft_image).to(self.args.device)
                    img_inputs = self.image_tokenizer(images=image, return_tensors="pt").pixel_values
                    return img_inputs, fft_image, sentiment_output, text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], label
                else:
                    logger.info(f"Image {img_path} does not exist. Skipping.")
                    return None 
            except Exception as e:
                logger.info(f"Error loading image {img_path}: {e}")
                return None
        else:
            text_feature = None
            try:
                if os.path.exists(img_path) and os.path.isfile(img_path):
                    fft_image = extract_spectral_features(img_path,output_dim=(128, 128))
                    fft_image = torch.tensor(fft_image).to(self.args.device)
                    text_feature,img_feature =  self.clip_encoder.get_features(text,img_path)
                    return img_feature, fft_image, sentiment_output, text_feature, label
                else:
                    logger.info(f"Image {img_path} does not exist. Skipping.")
                    return None 
            except Exception as e:
                logger.info(f"Image {img_path} does not exist. Skipping.")
                return None 
            

def custom_collate_fn_mmdata(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return [], []
    if len(valid_batch[0])> 5: 
        batch_images = torch.stack([item[0] for item in valid_batch])
        fft_images = torch.stack([item[1] for item in valid_batch])
        sentiment_outputs = torch.stack([item[2] for item in valid_batch])
        text_input_ids = torch.stack([item[3] for item in valid_batch])
        text_token_type_ids = torch.stack([item[4] for item in valid_batch])
        text_attention_mask = torch.stack([item[5] for item in valid_batch])
        batch_labels = torch.tensor([item[7] for item in valid_batch])
        return batch_images, fft_images, sentiment_outputs, text_input_ids, text_token_type_ids, text_attention_mask, batch_labels
    else: 
        batch_images = torch.stack([item[0] for item in valid_batch])
        fft_images = torch.stack([item[1] for item in valid_batch])
        sentiment_outputs = torch.stack([item[2] for item in valid_batch])
        text_features = torch.stack([item[3] for item in valid_batch])
        batch_labels = torch.stack([item[4] for item in valid_batch])
        return batch_images, fft_images, sentiment_outputs, text_features, batch_labels


def mm_data_loader(args):
    if args.dataset in ['Weibo17','Weibo21','CFND_dataset']:
        train_set = MMDataset(args, mode='train')
        valid_set = MMDataset(args, mode='val')
        test_set = MMDataset(args, mode='test')
        logger.info(f'Train Dataset: {len(train_set)}')
        logger.info(f'Valid Dataset: {len(valid_set)}')
        logger.info(f'Test Dataset: {len(test_set)}')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn_mmdata)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn_mmdata)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn_mmdata)
        return train_loader, valid_loader, test_loader
    else:
        logger.error(f'无效数据集: {args.dataset}，支持的数据集为 ["Weibo17", "Weibo21", "CFND_dataset"]')
        raise ValueError("无效数据集，请检查参数")


