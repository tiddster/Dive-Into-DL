import jieba
import pandas as pd
from wordcloud import WordCloud

# path = "F:\Dataset\Bilibili\\Math.csv"
# pic_path = "F:\Dataset\Bilibili\\Math.jpg"
path = "F:\Dataset\Bilibili\\Computer.csv"
pic_path = "F:\Dataset\Bilibili\\Computer_pasitive.jpg"

data = pd.read_csv(path, encoding='utf-8')
sentiList = list(data["情感"])
textList = list(data["文本"])

pos_text = ""
neg_text = ""
neu_text = ""

for senti, text in zip(sentiList, textList):
    if senti == "积极":
        pos_text += text
    elif senti == "消极":
        neg_text += text
    else:
        neu_text += text

ls = jieba.cut(pos_text)
text = ' '.join(ls)

stopwords = ["的","是","了","图片", "我", "宋浩","老师","宋","视频","王道","王","讲","看","有","郑"] # 去掉不需要显示的词

wc = WordCloud(font_path="msyh.ttc",
                         width = 1000,
                         height = 700,
                         background_color='white',
                         max_words=100,stopwords=stopwords)
# msyh.ttc电脑本地字体，写可以写成绝对路径
wc.generate(text) # 加载词云文本
wc.to_file(pic_path) # 保存词云文件
