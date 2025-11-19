import re
import pandas as pd
from model.text_encoder import TextEncoder
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from utils.process_data import preprocess_text

class TextDataset(Dataset):
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
        self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder).get_tokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.args.dataset in ['Weibo17','Weibo21']:
            text, label = self.df.iloc[index, 2:4 ].values
        elif self.args.dataset in ['CFND_dataset']:
            text, label = self.df.iloc[index, [1,3] ].values
            text = preprocess_text(text)
            text_tokens = self.text_tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True,
                                     padding='max_length', return_tensors="pt")
            return text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], label
        else:
            logger.error(f'无效数据集: {self.args.dataset}，支持的数据集为 ["Weibo17", "Weibo21", "CFND_dataset"]')
            raise ValueError("无效数据集，请检查参数")
        
def text_dataloader(args):
    if args.dataset in ['Weibo17','Weibo21','CFND_dataset']:
        train_set = TextDataset(args, mode='train')
        valid_set = TextDataset(args, mode='val')
        test_set = TextDataset(args, mode='test')
        logger.info(f'Train Dataset: {len(train_set)}')
        logger.info(f'Valid Dataset: {len(valid_set)}')
        logger.info(f'Test Dataset: {len(test_set)}')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4,
                        shuffle=True, pin_memory=False, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=4,
                        shuffle=False, pin_memory=False, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4,
                        shuffle=False, pin_memory=False, drop_last=True)
        return train_loader, valid_loader, test_loader
    else:
        logger.error(f'无效数据集: {args.dataset}，支持的数据集为 ["Weibo17", "Weibo21", "CFND_dataset"]')
        raise ValueError("无效数据集，请检查参数")

