import time

from aip import AipNlp
import pandas as pd
import json
import BilibiliComments
from threading import Lock

save_path = BilibiliComments.save_path
save_txts = BilibiliComments.save_txts
# 百度情感分析api
api_path = "F:\Dataset\\baidu_api.txt"
root_path = "F:\Dataset\\Bilibili\\"

# 医学
# path = "F:\Dataset\Bilibili\\Medical.json"
#
# fileM = open(path, 'r', encoding="utf-8")
# jsM = fileM.read()
# dicts = json.loads(jsM)
#
# textList = []
#
# for dict in dicts:
#     for _, value in dict.items():
#         print(value)
#         textList.append(value)
# print(textList)

# 经济社会学
path = "F:\Dataset\\Bilibili\\MentalBasic.csv"
data = pd.read_csv(path)
print(data)
textList = list(data["1"])
print(textList)

# 教资数据集
# path = "F:\Dataset\\Bilibili\\EQbasic.csv"
# data = pd.read_csv(path)
# textList = data["内容"]
# print(textList)

# 百度api
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


# 数学和计算机数据集
# textDic = {"AM": [], "LA": [], "PS": [], "CN": [], "COS": [], "CC": []}
# for (title, _), txt in zip(textDic.items(), save_txts):
#     textDic[title] = load_data(txt)

# AMDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
# LADict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
# PSDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
# CNDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
# COSDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
# CCDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

csvTextList = []


def classifyText(text):
    if text is not None:
        if len(text) == 0: return
        text = text.encode('gbk', errors='ignore').decode('gbk').encode('utf-8').decode('utf-8')
        res = client.sentimentClassify(text)
        time.sleep(0.25)
        # res = {'text': '后来，我们还是上了同一所大学[doge]', 'items': [{'confidence': 0.88032, 'negative_prob': 0.0538557, 'positive_prob': 0.946144, 'sentiment': 2}], 'log_id': 1572567722348422346}
        if res.get("items") != None:
            # print(res)
            itemsDict = res['items'][0]
            confidence = itemsDict['confidence']
            neg_prob = itemsDict['negative_prob']
            pos_prob = itemsDict['positive_prob']

            if confidence < 0.5:
                csvTextList.append(["中性", text])
            elif pos_prob > 0.5:
                csvTextList.append(["积极", text])
            elif neg_prob > 0.5:
                csvTextList.append(["消极", text])


# textList = ""
csvName = "Mental.csv"

i = 0
for text in textList:
    classifyText(text)
    i += 1
    if i % 100 == 0:
        print(len(csvTextList))
    if i == 3000:
        break

idx = pd.date_range(start=1, periods=len(csvTextList))
csvData = pd.DataFrame(csvTextList, columns=["情感", "文本"])
csvData.to_csv(root_path + csvName, encoding='utf-8')

# def save_dict(dict, filename):
#     js = json.dumps(dict)
#     file = open(root_path + filename, 'w')
#     file.write(js)
#     file.close()

# classifyDict(AMDict, textDic["AM"])
# save_dict(AMDict, "AMDict.txt")
# print(1)
# classifyDict(LADict, textDic["LA"])
# save_dict(LADict, "LADict.txt")
# print(1)
# classifyDict(PSDict, textDic["PS"])
# save_dict(PSDict, "PSDict.txt")
# print(1)
# classifyDict(CNDict, textDic["CN"])
# save_dict(CNDict, "CNDict.txt")
# print(1)
# classifyDict(COSDict, textDic["COS"])
# save_dict(COSDict, "COSDict.txt")
# print(1)
# classifyDict(CCDict, textDic["CC"])
# save_dict(CCDict, "CCDict.txt")

# for title, list in AMDict.items():
#     print(f"{title}: {len(list)}")
