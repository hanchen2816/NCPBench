import re
import random
import json  
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
import os
from test_LLM.common.llm_utils import LLM, get_response

def replace(number_list, sentence):
    numbers = re.findall(r'\d+', sentence)
    numbers = [int(num) for num in numbers]
    if number_list:
        for num in numbers:
            if number_list:
                random_number = random.choice(number_list)
                sentence = re.sub(str(num), str(random_number), sentence, 1)
        return sentence
    return None

def judge(question, assistant_output, is_chinese=True):
    numbers = re.findall(r'\d+', question)
    numbers = [int(num) for num in numbers]
    target_count = numbers[0]
    
    if is_chinese:
        actual_count = len(re.findall(r'[\u4e00-\u9fff]', assistant_output))
    else:
        actual_count = len(re.findall(r'\b\w+\b', assistant_output))
    
    return actual_count == target_count

def number_lists(type, is_chinese=True, data_dir=None):
    try:
        path = os.path.join(data_dir, 'number.json') if data_dir else ('data/chinese/number.json' if is_chinese else 'data/english/number.json')
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data[type]
    except Exception as e:
        print(f"读取数字列表时发生错误：{e}")
        return []

def test(type, is_chinese=True, model_name=None, api_key=None, data_dir=None, output_dir=None, base_url=None):
    # 设置默认值
    if model_name is None:
        model_name = "Gpt-4omini" if is_chinese else "Gpt-4omini2"
    if data_dir is None:
        data_dir = 'data/chinese' if is_chinese else 'data/english'
    if output_dir is None:
        output_dir = 'result/words/中文' if is_chinese else 'result/words/英文'
    
    # 构建文件路径
    path = os.path.join(output_dir, '得分', f'"{model_name}"-{type}测试结果.json')
    outpath = os.path.join(output_dir, '输出', f'"{model_name}"-{type}模型输出结果.json')
    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    total = true = false = 0
    
    # 读取已处理的结果
    if os.path.exists(outpath):
        with open(outpath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if item["判断"]:
                        true += 1
                    else:
                        false += 1
                    total += 1
                except:
                    continue
    
    # 读取任务文件
    task_file = os.path.join(data_dir, '简单生成任务.txt' if is_chinese else 'generate_Q.txt')
    try:
        with open(task_file, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
    except Exception as e:
        print(f"读取任务文件出错: {e}")
        return
    
    # 从断点处继续处理
    start_index = sum(1 for _ in open(outpath, 'r', encoding='utf-8')) if os.path.exists(outpath) else 0
    lines = lines[start_index:]
    
    # 处理任务
    for sentence in tqdm(lines):
        number_list = number_lists(type, is_chinese, data_dir)
        assistant_input = replace(number_list, sentence)
        if not assistant_input:
            continue
            
        assistant_output = LLM(assistant_input, is_chinese, model_name, api_key, base_url)
        
        sample = {
            "问题": assistant_input,
            "回答": assistant_output,
            "模型生成个数": len(re.findall(r'[\u4e00-\u9fff]' if is_chinese else r'\b\w+\b', assistant_output)),
            "判断": judge(assistant_input, assistant_output, is_chinese)
        }
        
        try:
            with open(outpath, 'a', encoding='utf-8') as f:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"写入结果时出错: {e}")
            continue
        
        total += 1
        if sample["判断"]:
            true += 1
        else:
            false += 1
    
    # 保存统计结果
    if total > 0:
        stats = {
            f"生成{type}个数成功率": true / total,
            f"生成{type}个数失败率": false / total
        }
        try:
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"写入统计结果时出错: {e}")

if __name__ == '__main__':
    number_types = ["素数", "奇数", "偶数", "2的幂", "非2的幂", "完全平方数", "阶乘", "质因数"]
    
    # 测试中文
    for type in number_types:
        test(type, is_chinese=True)
    
    # 测试英文
    for type in number_types:
        test(type, is_chinese=False)
