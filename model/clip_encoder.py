import torch 
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name

class ClipEncoder():
    def __init__(self, args, pretrained_dir):
        self.args = args
        self.model,  self.preprocess = load_from_name("ViT-L-14", device=self.args.device, download_root=pretrained_dir)
        self.model.eval()
        
        
    def get_features(self,text,image_path):
        preprocessed_image = self.preprocess(Image.open(image_path))
        image_tensor = torch.tensor(preprocessed_image).to(self.args.device)
        image = image_tensor.unsqueeze(0)
        text = clip.tokenize([text]).to(self.args.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True) 
            text_features /= text_features.norm(dim=-1, keepdim=True)    
        return text_features, image_features
    
    def get_entity_features(self,entity_list):
        entity_list = clip.tokenize(entity_list).to(self.args.device)
        with torch.no_grad():
            entity_features = self.model.encode_text(entity_list)
            entity_features /= entity_features.norm(dim=-1, keepdim=True)    
        return entity_features