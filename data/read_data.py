import json
from tqdm import tqdm

jsonl_file_path = 'common_zh_70k.jsonl'

results = []
# 打开JSON Lines文件
with open(jsonl_file_path, 'r', encoding='utf-8') as file:
    # 逐行读取文件内容
    for line in tqdm(file):
        # 解析JSON行
        json_object = json.loads(line.strip())
        
        # 处理json_object，根据需要执行操作
        #print(json_object['conversation'])
        #print(len(json_object['conversation']))
        #print(json_object['conversation'][0])

        if len(json_object['conversation'])>=2:
            rr = []
            for cc in range(len(json_object['conversation'])-1):
                rr.append([str(json_object['conversation'][cc]['human']), str(json_object['conversation'][cc]['assistant'])])

            info = {
                "instruction": str(json_object['conversation'][-1]['human']),
                "input": "",
                "output": str(json_object['conversation'][-1]['assistant']),
                "history": rr
              }
            results.append(info)
            
        if len(json_object['conversation'])==1:
            info = {
                "instruction": str(json_object['conversation'][0]['human']),
                "input": "",
                "output": str(json_object['conversation'][0]['assistant']),
                "history": []
              }
            results.append(info)
        
        # 打印完第一行后终止循环
        #break

with open('./sharegpt-70k.json', 'w', encoding="utf-8") as f1:
    json.dump(results, f1, ensure_ascii=False, indent=4)
