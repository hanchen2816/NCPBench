import re
import random
import json  
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
import os
from test_LLM.common.llm_utils import LLM, get_response

def replace_range(sentence):
    """替换句子中的数字范围为随机数"""
    # 匹配形如"1-10"或"1到10"的范围
    range_pattern = r'(\d+)[-到](\d+)'
    numbers = re.findall(range_pattern, sentence)
    
    if numbers:
        for start, end in numbers:
            start = int(start)
            end = int(end)
            if start <= end:
                random_number = random.randint(start, end)
                sentence = re.sub(f'{start}[-到]{end}', str(random_number), sentence)
        return sentence
    return None


def judge(question, assistant_output, is_chinese=True):
    """判断生成的文本长度是否在要求范围内"""
    # 从问题中提取长度范围
    if is_chinese:
        length_pattern = r'（不超过\s*(\d+)\s*词，不低于\s*(\d+)\s*词）'
    else:
        length_pattern = r'\(no more than\s*(\d+)\s*words,\s*no less than\s*(\d+)\s*words\)'
    
    length_ranges = re.findall(length_pattern, question)
    
    if not length_ranges:
        return False
        
    max_length = int(length_ranges[0][0])
    min_length = int(length_ranges[0][1])
    
    # 计算实际长度
    if is_chinese:
        actual_length = len(re.findall(r'[\u4e00-\u9fff]', assistant_output))
    else:
        actual_length = len(re.findall(r'\b\w+\b', assistant_output))
        
    # 检查长度是否在范围内
    return min_length <= actual_length <= max_length

def test(is_chinese=True, model_name=None, api_key=None, data_dir=None, output_dir=None, base_url=None):
    """运行范围约束测试"""
    # 设置默认值
    if model_name is None:
        model_name = "qwen1.5-72b-chat"
    if data_dir is None:
        data_dir = 'data/range_Q'
    if output_dir is None:
        output_dir = os.path.join('result/range_test', '中文' if is_chinese else '英文')
    
    # 获取当前日期
    today = datetime.now()
    formatted_date = today.strftime("%m.%-d").lstrip("0")
    
    # 构建文件路径
    path = os.path.join(output_dir, '得分', f'{formatted_date}-{model_name}-范围测试结果.json')
    outpath = os.path.join(output_dir, '输出', f'{formatted_date}-{model_name}-范围测试输出结果.json')
    
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
    task_file = os.path.join(data_dir, 'CH.txt' if is_chinese else 'EN.txt')
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
        assistant_input = sentence
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
            "范围约束成功率": true / total,
            "范围约束失败率": false / total
        }
        try:
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"写入统计结果时出错: {e}")

if __name__ == '__main__':
    # 测试中文
    test(is_chinese=True)
    
    # 测试英文
    test(is_chinese=False)
