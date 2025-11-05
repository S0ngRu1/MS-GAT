
# encoding:utf-8

import requests
import base64
import pandas as pd
import json
import time
from tqdm import tqdm
import sys


def read_json(path: str):
    with open(path, 'r') as f:
        data = json.loads(f.read())
    return data
def contains_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff' or \
           '\u3400' <= char <= '\u4dbf' or \
           '\u20000' <= char <= '\u2a6df' or \
           '\u2a700' <= char <= '\u2b73f' or \
           '\u2b740' <= char <= '\u2b81f' or \
           '\u2b820' <= char <= '\u2ceaf' or \
           '\u2ceb0' <= char <= '\u2ebef' or \
           '\u30000' <= char <= '\u3134f':
            return True
    return False




def advanced_general():
    mode = 'val'
    
    df1 = pd.read_csv(f"datasets/Weibo21/{mode}.csv")
    ocr_datas = read_json(f"datasets/Weibo21/ocr_data_{mode}.json")
    obj_datas = []
    # obj_datas = read_json(f"datasets/Weibo21/obj_data_{mode}.json")
    
    # 获取已处理的最大num
    last_num = 0  # 默认起始值
    if obj_datas:
        last_num = 0
    
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"

    for index, row in df1.iterrows():
        if index <= last_num:
            continue
        try:    
            if not contains_chinese(ocr_datas[index]['plain_text']):
                print(f"正在处理index: {index}, num: {row['tweet_id']}")
                image_path = row['image_name']
                image_file = f'datasets/Weibo21/new_images/{image_path}'
                f = open(image_file, 'rb')
                img = base64.b64encode(f.read())
                access_token = "24.23565ba56fb5b60c676d5f4d56c1254c.2592000.1748916947.282335-118726748"
                params = {"image":img}
                request_url = request_url + "?access_token=" + access_token
                headers = {'content-type': 'application/x-www-form-urlencoded'}
                session = requests.Session()
                response = session.post(request_url, data=params, headers=headers)
                if response:
                    response = response.json()
                    obj_results = response.get('result','')
                    session.close()
                    if obj_results:
                        obj_results = [item for item in obj_results if item['score'] > 0.5]
                        obj_results = [item['keyword'] for item in obj_results]
                        print(obj_results)
                        obj_datas.append({
                            "num": row['tweet_id'],
                            "title": row['tweet_content'],
                            "obj_list": obj_results
                        })
            else:
                print(f"skip:{index}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"index{index}")
            with open(f'datasets/Weibo21/obj_data_{mode}.json', 'w', encoding='utf-8') as f:
                json.dump(obj_datas, f, ensure_ascii=False, indent=4)
            print("程序遇到错误，准备退出。")
            sys.exit()
    with open(f'datasets/Weibo21/obj_data_{mode}.json', 'w', encoding='utf-8') as f:
        json.dump(obj_datas, f, ensure_ascii=False, indent=4)
        
        

def advanced_general_single():
    image_path = 'datasets/CFND_dataset/images/real_news_image/362.jpg'
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
    image_file = f'{image_path}'
    f = open(image_file, 'rb')
    img = base64.b64encode(f.read())
    access_token = "24.31136111035be2c5321db26089b8a6e1.2592000.1745845649.282335-118279296"
    params = {"image":img}
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    session = requests.Session()
    response = session.post(request_url, data=params, headers=headers)
    if response:
        response = response.json()
        obj_results = response.get('result','')
        session.close()
        if obj_results:
            obj_results = [item for item in obj_results if item['score'] > 0.5]
            obj_results = [item['keyword'] for item in obj_results]
            print(obj_results)
        
    

if __name__ == '__main__':
    advanced_general()
