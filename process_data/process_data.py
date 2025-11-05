import json

if __name__ == '__main__':
    processed_datas = []

    mode = 'train'
    output_path = f'datasets/Weibo21/processed_data_{mode}.json'
    ocr_data = json.load(open(f'datasets/Weibo21/ocr_data_{mode}.json', 'r', encoding='utf-8'))
    obj_data = json.load(open(f'datasets/Weibo21/obj_data_{mode}.json', 'r', encoding='utf-8'))
    ner_data = json.load(open(f'datasets/Weibo21/ner_data_{mode}.json', 'r', encoding='utf-8'))
    ner_data_dict = {}
    obj_data_dict = {}
    for i in range(len(obj_data)):
        obj_data_dict[obj_data[i]["num"]] = obj_data[i]
    for i in range(len(ner_data)):
        ner_data_dict[ner_data[i]["index"]] = ner_data[i]

    for i in range(len(ocr_data)):
        processed_data = ocr_data[i]
        processed_data["title_ner"] = []
        processed_data["plain_text_ner"] = []
        if i in ner_data_dict:
            processed_data["title_ner"] = set([*ner_data_dict[i]["title_ner"]])
            processed_data["title_ner"] = list(processed_data["title_ner"])
        if i in ner_data_dict:
            processed_data["plain_text_ner"] = set([*ner_data_dict[i]["plain_text_ner"]])
            processed_data["plain_text_ner"] = list(processed_data["plain_text_ner"])
        if processed_data["num"] in obj_data_dict:
            processed_data["obj_list"] = obj_data_dict[processed_data["num"]]["obj_list"]

        processed_datas.append(processed_data)

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(processed_datas, f, indent=4, ensure_ascii=False)