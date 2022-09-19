import requests
import time

save_path = "F:\Dataset\Bilibili\\"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36 Edg/99.0.1150.39'
}

init_urls = [
    f"https://api.bilibili.com/x/v2/reply/main?csrf=7deae47411e37406ef07209671282de6&mode=3&next=0&oid=48624233&plat=1&seek_rpid=&type=1",
    f"https://api.bilibili.com/x/v2/reply/main?csrf=7deae47411e37406ef07209671282de6&mode=3&next=0&oid=29971113&plat=1&type=1"
    f"https://api.bilibili.com/x/v2/reply/main?csrf=7deae47411e37406ef07209671282de6&mode=3&next=0&oid=36206436&plat=1&type=1"
]
save_txts = ["advance-math.txt", "LinearAlgebra.txt","ProbabilityStatistics.txt"]


def getdata(save_txt, init_url):
    page = 0
    url = init_url
    with open(save_path + save_txt, 'w', encoding='utf-8') as f:
        while True:
            #print(f"==============={page}==================")
            response = requests.get(url, headers=headers, verify=False)
            response.encoding = 'utf-8'
            Json_reply_list = response.json()['data']['replies']
            if Json_reply_list:
                for i in range(len(Json_reply_list)):
                    comment = Json_reply_list[i]["content"]["message"]
                    #print(comment)
                    f.write(comment + "\n")
            else:
                print("======================over=====================")
                break
            page += 1
            url = url.replace(f"next={page - 1}", f"next={page}")
            print(url)


for txt, url in zip(save_txts, init_urls):
    getdata(txt, url)
