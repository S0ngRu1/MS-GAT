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



# TODO: 多模态数据集类  
class MMDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.save = []
        self.max_length = 256
        if args.dataset in ['Weibo17','Weibo21']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '.csv', encoding='utf-8').dropna()
        elif args.dataset in ['CFND_dataset']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '_data.csv', encoding='utf-8').dropna()

        else:
            logger.info('数据集无效')
            return
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
        fft_imge = None
        if self.args.dataset in ['Weibo17','Weibo21']:
            _, image_name, text, label = self.df.iloc[index].values
            img_path = self.args.data_dir +'/'+ self.args.dataset +'/new_images/' + image_name
            text = preprocess_text(text)
        
        elif self.args.dataset in ['CFND_dataset']:
            _, title, image, label = self.df.iloc[index].values
            img_path = os.path.join(self.args.data_dir,self.args.dataset, image)
            text = preprocess_text(title)
            
        
        sentiment_output = sentiment_predict(text, self.sentiment_config, self.sentiment_model)
        if self.args.method not in ['FND-2-CLIP']:
            text_tokens = self.text_tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True,
                                        padding='max_length', return_tensors="pt")
            original_shape = (1, 3, 224, 224)
            zero_img_inputs = torch.zeros(original_shape)
            try:
                if os.path.exists(img_path) and os.path.isfile(img_path):
                    image = Image.open(os.path.join(img_path)).convert("RGB")
                    image = self.transforms(image)
                    fft_imge = extract_spectral_features(img_path,output_dim=(128, 128))
                    fft_imge = torch.tensor(fft_imge).to(self.args.device)
                    img_inputs = self.image_tokenizer(images=image, return_tensors="pt").pixel_values
                else:
                    img_inputs = zero_img_inputs
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                img_inputs = zero_img_inputs
            return img_inputs, fft_imge, sentiment_output, text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], label
        
        
        else:
            text_feature = None
            try:
                if os.path.exists(img_path) and os.path.isfile(img_path):
                    fft_imge = extract_spectral_features(img_path,output_dim=(128, 128))
                    fft_imge = torch.tensor(fft_imge).to(self.args.device)
                    text_feature,img_feature =  self.clip_encoder.get_features(text,img_path)
                else:
                    img_feature = torch.randn((1, 768))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                img_feature = torch.randn((1, 768))
            return img_feature, fft_imge, sentiment_output, text_feature, label
        