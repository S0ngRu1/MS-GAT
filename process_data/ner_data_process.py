from pprint import pprint
from paddlenlp import Taskflow

schema = ['时间', '地点', '人物',"事件","主体", '物体' ,"国家", '动作', "行为","状态","民族"] 
# schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
ie = Taskflow('information_extraction',
              schema= ['时间', '选手', '赛事名称'],
              schema_lang="zh",
              batch_size=1,
              precision='float16')
pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))