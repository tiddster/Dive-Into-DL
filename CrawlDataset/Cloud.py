import jieba
import pandas as pd
from wordcloud import WordCloud

# path = "F:\Dataset\Bilibili\\Math.csv"
# pic_path = "F:\Dataset\Bilibili\\Math.jpg"
path = "F:\Dataset\Bilibili\\Medical.csv"
pos_path = "F:\Dataset\Bilibili\\positive.jpg"
neg_path = "F:\Dataset\Bilibili\\negative.jpg"
neu_path = "F:\Dataset\Bilibili\\neutral.jpg"
medical = "F:\Dataset\Bilibili\\medical.jpg"

data = pd.read_csv(path, encoding='utf-8')
print(data)
sentiList = list(data["情感"])
textList = list(data["文本"])

i, j, k = 0, 0, 0
pos_text = ""
neg_text = ""
neu_text = ""
textM = ""

for senti, text in zip(sentiList, textList):
    textM += text
    if senti == "积极":
        pos_text += str(text)
        i += 1
    elif senti == "消极":
        neg_text += str(text)
        j += 1
    else:
        neu_text += str(text)
        k += 1

print(len(textList))
print(i/len(textList))
print(j/len(textList))
print(k/len(textList))



pos_ls = jieba.cut(pos_text)
neg_ls = jieba.cut(neg_text)
neu_ls = jieba.cut(neu_text)
ls = jieba.cut(textM)
words = ' '.join(ls)
pos_words = ' '.join(pos_ls)
neg_words = ' '.join(neg_ls)
neu_words = ' '.join(neu_ls)

stopwords = ["的","是","了","图片", "我", "宋浩","老师","宋","视频","王道","王","讲","看","有","郑","课程","吗","啊","这个"] # 去掉不需要显示的词

wc = WordCloud(font_path="msyh.ttc",
                         width = 1000,
                         height = 700,
                         background_color='white',
                         max_words=100,stopwords=stopwords)
# msyh.ttc电脑本地字体，写可以写成绝对路径
wc.generate(words)
wc.to_file(medical)

wc.generate(pos_words)
wc.to_file(pos_path)
wc.generate(neg_words)
wc.to_file(neg_path)
wc.generate(neu_words)
wc.to_file(neu_path)
