import requests
import time

save_path = "F:\Dataset\Bilibili\\"

page = 0
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36 Edg/99.0.1150.39'
}

with open(save_path + "math.txt", 'w') as f:
    while True:
        print(f"==============={page}==================")
        url = f"https://api.bilibili.com/x/v2/reply/main?csrf=05e21359595373aab5f2d891d9ab431d&mode=3&next={page}&oid=471900826&plat=1&type=1"
        response = requests.get(url, headers=headers, verify=False)
        response.encoding = 'utf-8'
        Json_reply_list = response.json()['data']['replies']
        if Json_reply_list:
            for i in range(len(Json_reply_list)):
                comment = Json_reply_list[i]["content"]["message"]
                f.write(comment + "\n")
                print(comment)
        else:
            print("======================over=====================")
            break
        page += 1


