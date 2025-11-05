import torch 
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

class ClipEncoder():
    def __init__(self, args, pretrained_dir):
        self.args = args
        self.model,  self.preprocess = load_from_name("ViT-L-14", device=self.args.device, download_root=pretrained_dir)
        self.model.eval()
        
        
    def get_features(self,text,image_path):
        image =  self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.args.device)
        text = clip.tokenize([text]).to(self.args.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
            image_features /= image_features.norm(dim=-1, keepdim=True) 
            text_features /= text_features.norm(dim=-1, keepdim=True)    
        return text_features, image_features
    
    def get_entity_features(self,entity_list):
        entity_list = clip.tokenize(entity_list).to(self.args.device)
        with torch.no_grad():
            entity_features = self.model.encode_text(entity_list)
            # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
            entity_features /= entity_features.norm(dim=-1, keepdim=True)    
        return entity_features