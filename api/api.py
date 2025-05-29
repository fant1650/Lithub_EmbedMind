import requests


def post(content):
    key = "sk-1101e28471c147b085d3ccecc8f016f2"
    url = "https://chat.ecnu.edu.cn/open/api/v1/chat/completions"
    headers = {
        "Authorization":f"{key}",
        "Content-Type":"application/json"
    }
    
    data = {
        "model": "ecnu-max",
        "messages": [
            {"role": "system", "content": "你是一个提取关键词的模型"},
            {"role": "user", "content": f"提取以下文档关键词：'{content}',输出关键词用空格隔开，不要输出任何无关文字"}
        ],
        "search_mode":"enable"
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']

content = "Mitochondria retain bacterial traits due to their endosymbiotic origin, but host cells do not recognize them as foreign because the organelles are sequestered. However, the regulated release of mitochondrial factors into the cytosol can trigger cell death, innate immunity and inflammation. This selective breakdown in the 2-billion-year-old endosymbiotic relationship enables mitochondria to act as intracellular signalling hubs. Mitochondrial signals include proteins, nucleic acids, phospholipids, metabolites and reactive oxygen species, which have many modes of release from mitochondria, and of decoding in the cytosol and nucleus. Because these mitochondrial signals probably contribute to the homeostatic role of inflammation, dysregulation of these processes may lead to autoimmune and inflammatory diseases. A potential reason for the increased incidence of these diseases may be changes in mitochondrial function and signalling in response to such recent phenomena as obesity, dietary changes and other environmental factors. Focusing on the mixed heritage of mitochondria therefore leads to predictions for future insights, research paths and therapeutic opportunities. Thus, whereas mitochondria can be considered 'the enemy within' the cell, evolution has used this strained relationship in intriguing ways, with increasing evidence pointing to the recent failure of endosymbiosis being critical for the pathogenesis of inflammatory diseases."

print(post(content))
