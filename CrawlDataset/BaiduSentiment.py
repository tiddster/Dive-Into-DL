import time

from aip import AipNlp
import json
import BilibiliComments

save_path = BilibiliComments.save_path
save_txts = BilibiliComments.save_txts
# 百度情感分析api
api_path = "F:\Dataset\\baidu_api.txt"
root_path = "F:\Dataset\\Bilibili"

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

AMDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
LADict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
PSDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
CNDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
COSDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}
CCDict = {"pos": [], "neg": [], "neu": [], "low_confidence": []}

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)


def classifyDict(dict, textList):
    for text in textList:
        if len(text) == 0: continue
        text = text.encode('gbk', errors='ignore').decode('gbk').encode('utf-8').decode('utf-8')
        res = client.sentimentClassify(text)
        time.sleep(1)
        # res = {'text': '后来，我们还是上了同一所大学[doge]', 'items': [{'confidence': 0.88032, 'negative_prob': 0.0538557, 'positive_prob': 0.946144, 'sentiment': 2}], 'log_id': 1572567722348422346}
        if res.get("items") != None:
            print(res)
            itemsDict = res['items'][0]
            confidence = itemsDict['confidence']
            neg_prob = itemsDict['negative_prob']
            pos_prob = itemsDict['positive_prob']

            if confidence < 0.5:
                dict["low_confidence"].append(text)
                if 0.4 <= pos_prob <= 0.6:
                    dict["neu"].append(text)
            elif pos_prob >= 0.5:
                dict["pos"].append(text)
            elif neg_prob > 0.5:
                dict["pos"].append(text)


classifyDict(AMDict, textDic["AM"])
print(1)
classifyDict(LADict, textDic["LA"])
print(1)
classifyDict(PSDict, textDic["PS"])
print(1)
classifyDict(CNDict, textDic["CN"])
print(1)
classifyDict(COSDict, textDic["COS"])
print(1)
classifyDict(CCDict, textDic["CC"])


def save_dict(dict, filename):
    js = json.dumps(dict)
    file = open(root_path + filename, 'w')
    file.write(js)
    file.close()


save_dict(AMDict, "AMDict.txt")
save_dict(LADict, "LADict.txt")
save_dict(PSDict, "PSDict.txt")
save_dict(CNDict, "CNDict.txt")
save_dict(COSDict, "COSDict.txt")
save_dict(CCDict, "CCDict.txt")

for title, list in AMDict.items():
    print(f"{title}: {len(list)}")
