from hanlp_restful import HanLPClient
from image_detection import contains_chinese, read_json
import json
import time
#zh中文，mul多语种
HanLP = HanLPClient('https://www.hanlp.com/api', auth="ODA3M0BiYnMuaGFubHAuY29tOklkRndyRlRTSDFsUmd4UWI=", language='zh')
ocr_datas = read_json("datasets/Weibo21/ocr_data_train.json")
request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
# 二进制方式打开图片文件
results = []
for index, ocr_data in enumerate(ocr_datas):
    try:
        print(f"正在处理index: {index}")  
        result = HanLP.parse(ocr_data['title'], tasks='ner/msra')
        title_ner = [entity[0] for entity in result['ner/msra'][0]]
        plain_text_ner = []
        if contains_chinese(ocr_data['plain_text']):
            result = HanLP.parse(ocr_data['plain_text'], tasks='ner/msra')
            plain_text_ner = [entity[0] for entity in result['ner/msra'][0]]
        
        results.append({
                "index": index,
                "title_ner": title_ner,
                "plain_text_ner": plain_text_ner
            })
        time.sleep(2)
    except Exception as e:
        print(f"Error: {e}")
        continue

# 将所有结果保存为 JSON 文件
with open('datasets/Weibo21/ner_data_train.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

