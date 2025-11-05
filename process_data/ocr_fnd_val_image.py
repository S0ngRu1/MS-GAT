#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CFND 
@File    ：ocr_demo.py
@IDE     ：PyCharm 
@Author  ：Cai Songrui 
@Date    ：2025/2/22 11:15 
'''
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import json
tokenizer = AutoTokenizer.from_pretrained('process_data/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('process_data/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()
# 读取 CSV 文件
df = pd.read_csv("datasets/Weibo21/val.csv") 
result = []
for index, row in df.iterrows():
    print(f"正在处理index: {index}")
    image_path = row['image_name']
    image_file = f'datasets/Weibo21/new_images/{image_path}'

    # plain texts OCR
    res = model.chat(tokenizer, image_file, ocr_type='ocr')

    result.append({
        "num": row['tweet_id'],
        "title": row['tweet_content'],
        "plain_text": res
    })

# 将所有结果保存为 JSON 文件
with open('datasets/Weibo21/ocr_data_val.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)


