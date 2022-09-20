from aip import AipNlp
import json
import BilibiliComments

save_path = BilibiliComments.save_path
save_txts = BilibiliComments.save_txts
# 百度情感分析api
api_path = "F:\Dataset\\baidu_api.txt"

file = open(api_path, 'r')
js = file.read()
dic = json.loads(js)
APP_ID = dic['APP_ID']
API_KEY = dic['API_KEY']
SECRET_KEY = dic['SECRET_KEY']


# 加载数据集
def load_data(txt):
    with open(save_path + txt, "r", encoding="utf-8") as f:
        textList = f.read().split('\n')
    return textList


textDic = {"AM": [], "LA": [], "PS": [], "CN": [], "COS": [], "CC": []}
for (title, _), txt in zip(textDic.items(), save_txts):
    textDic[title] = load_data(txt)

AMDict = {"pos":[], "neg":[], "low_confidence":[]}
LADict = {"pos":[], "neg":[], "low_confidence":[]}
PSDict = {"pos":[], "neg":[], "low_confidence":[]}
CNDict = {"pos":[], "neg":[], "low_confidence":[]}
COSDict = {"pos":[], "neg":[], "low_confidence":[]}
CCDict = {"pos":[], "neg":[], "low_confidence":[]}

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
# res = client.sentimentClassify()
# print(res)
