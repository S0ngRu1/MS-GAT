# -*- coding: utf-8 -*-
# Author: caisongrui
# Date: 2025-03-24
# Description: Description of the file


import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import os
import shutil
from model.model import MyModel
from model.text_encoder import TextEncoder
from model.image_encoder import ImageEncoder
from data.dataloader import preprocess_text


def load_model(args, checkpoint_path):
    """加载预训练模型"""
    model = MyModel(args)
    if torch.cuda.is_available():
        model = model.cuda()
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def get_transforms():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
        ]
    )

def predict_single_sample(model, text, image_path, args):
    """预测单个文本和图片对
    
    Args:
        model: 加载好的模型
        text (str): 输入文本
        image_path (str): 图片路径
        args: 模型参数配置
    
    Returns:
        prediction (int): 预测的类别 (0 或 1)
        probability (float): 预测的概率值
    """
    # 初始化tokenizer
    text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, 
                               text_encoder=args.text_encoder).get_tokenizer()
    image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir, 
                                 image_encoder=args.image_encoder).get_tokenizer()
    
    # 文本预处理
    text = preprocess_text(text)
    text_tokens = text_tokenizer(text, 
                               max_length=256,
                               add_special_tokens=True,
                               truncation=True,
                               padding='max_length',
                               return_tensors="pt")
    
    # 图像预处理
    transforms = get_transforms()
    try:
        if os.path.exists(image_path) and os.path.isfile(image_path):
            image = Image.open(image_path).convert("RGB")
            image = transforms(image)
            img_inputs = image_tokenizer(images=image, return_tensors="pt").pixel_values
        else:
            img_inputs = torch.zeros((1, 3, 224, 224))
    except OSError as e:
        print(f"Error loading image {image_path}: {e}")
        img_inputs = torch.zeros((1, 3, 224, 224))
    
    # 将数据移到GPU（如果可用）
    if torch.cuda.is_available():
        text_input_ids = text_tokens['input_ids'].cuda()
        text_token_type_ids = text_tokens['token_type_ids'].cuda()
        text_attention_mask = text_tokens['attention_mask'].cuda()
        img_inputs = img_inputs.cuda()
    
    # 模型推理
    with torch.no_grad():
        text = (text_input_ids, text_token_type_ids, text_attention_mask)
        logits = model.infer(text, img_inputs, None)
        probs = F.softmax(logits, dim=1)
        
    prediction = torch.argmax(probs, dim=1).item()
    probability = probs[0][prediction].item()
    
    return prediction, probability


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.pretrained_dir = 'Pretrained'
            self.text_encoder = "bert-base-chinese"
            self.image_encoder = "vit-base"
            self.method = "CMAWSC"

    args = Args()

    # 加载模型
    model = load_model(args, "results/models/Weibo21-CMAWSC—new_1-best.pth")
    import pandas as pd
    test_datasets_path = "datasets/Weibo21/test.csv"
    test_datas = pd.read_csv(test_datasets_path, encoding='utf-8')
    results = []
    for i in range(len(test_datas)):
        print(f"Processing {i+1}/{len(test_datas)}")
        id = test_datas['tweet_id'][i]
        text = test_datas['tweet_content'][i]
        
        image_name = test_datas['image_name'][i]
        if pd.isnull(image_name): 
            continue
        label = '虚假' if test_datas['label'][i] == 1 else '真实'
        image_path = os.path.join("datasets/Weibo21/new_images", image_name)
        prediction, probability = predict_single_sample(model, text, image_path, args)
        prediction = '虚假' if prediction == 1 else '真实'
        results.append({"id": str(id),"label": label, "claim": text,  "prediction": prediction, "domain":None, 'pic': image_path})
        # 复制图片到结果文件夹
        if not os.path.exists('prediction_results'):
            os.makedirs('prediction_results')
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join('prediction_results', image_name))
    
    # 保存结果
    with open('prediction.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # # 预测单个样本
    # text = "【扩散！别再传播谣言了！转发辟谣】今晚，你的朋友圈都被微信公开课PRO版的“我和微信的故事”刷屏了吗？然而，随之而来的是各种谣言：有的说这个应用会盗号，有的说会盗刷支付宝。对此，微信官方已经辟谣：链接不存在病毒或者木马，更不存在账号里的钱被盗的情况。O【辟谣】我和微信的故事是真的！盗号什么的都是谣言！"
    # image_path = "datasets/Weibo17/images/4e5b54d8gw1ezuskgvmolj20c808v3z5.jpg"
    # prediction, probability = predict_single_sample(model, text, image_path, args)
    # print(f"预测类别: {prediction}")
    # print(f"预测概率: {probability:.4f}")
