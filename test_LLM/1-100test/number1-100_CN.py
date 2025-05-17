import re
import random
import json  
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
import os


def replace_number(i, sentence):
    numbers = re.findall(r'\d+', sentence)  # 提取句子中的所有数字
    numbers = [int(num) for num in numbers]  # 转换为数字列表
    # 如果文件中的素数列表有效，进行替换
    if i:
        for num in numbers:
            # 确保number_list不为空
            if i:
                random_number = i  # 随机选择一个素数
                # 替换句子中的数字
                sentence = re.sub(str(num), str(random_number), sentence, 1)  # 只替换第一个匹配的数字

        print("替换后的句子：", sentence)
        return sentence
    else:
        print("素数列表为空或格式错误，无法进行替换。")

def get_response(messages):
    client = OpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key="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",
        # api_key="sk-4b2369448874441b8efe0befe39b9ba2",
        # # 填写DashScope服务的base_url
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    completion = client.chat.completions.create(
        # model="qwen1.5-72b-chat",
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0.8,
        top_p=0.8
    )
    return completion

def LLM(assistant_input):
    messages = [
        {'role': 'system', 'content': "你是一个智能助手，能够根据用户的需求生成内容，请严格遵守用户的要求，包括内容数量、质量、格式等规则，如果出现数字，要完全按照数字字数生成，不要多也不要少一个字。输出格式为一句话。"},
        {'role': 'user', 'content': "请根据下面要求，生成对应的文段，要求其中除去符号后，汉字的个数必须跟要求的一致"+str(assistant_input)}
    ]

    try:
        completion = get_response(messages)
        # 防御性编程：先检查completion里是否有choices、choices[0]、message等
        if not hasattr(completion, 'choices') or not completion.choices:
            print("模型返回结果中没有choices信息。")
            return "抱歉，模型未返回有效的内容。"
        
        # 取出真正内容
        assistant_output = completion.choices[0].message.content
        return assistant_output
    
    except IndexError as ie:
        # 如果completion.choices为空或者结构异常
        print(f"解析响应时发生IndexError: {ie}")
        return "抱歉，未能从模型获取到内容。"
    
    except Exception as e:
        # 捕获其他所有错误
        print(f"调用 get_response 或解析响应时发生异常: {e}")
        return "抱歉，发生了错误，无法完成请求。"

def judge(question, assistant_output):
    numbers = re.findall(r'\d+', question)  # 提取句子中的所有数字
    numbers = [int(num) for num in numbers]  # 转换为数字列表
    print("要求生成的字的个数："+str(numbers[0]))
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', assistant_output)
    chinese_count = len(chinese_characters)
    print(f"大模型生成的句子：{assistant_output}")
    print("大模型生成的字的个数："+ str(chinese_count))
    if chinese_count == numbers[0]:
        return True
    else:
        return False
    
def number_lists(type):
        # 要替换成的数字
        path = 'data/chinese/number.json'
        # 读取文件中的内容，确保文件包含一个有效的JSON格式列表
        with open(path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # 使用json加载列表数据
                number_list = data[type]
            except json.JSONDecodeError as e:
                print(f"读取文件时发生错误：{e}")
                number_list = []
        return number_list

def test(i):
    path = f'test_LLM/1-100test/gpt4omini_result/数字{i}测试结果.json'
    outpath = f'test_LLM/1-100test/gpt4omini_result/数字{i}模型输出结果.json'
    lines = []
    total = 0
    true=0
    false=0

    if os.path.exists(outpath):
        try:
            with open(outpath, 'r', encoding='utf-8') as f:
                data = [line.strip() for line in f]  # 去掉每行末尾的换行符
            total = len(data)
            for item_str in data:
                try:
                    item_dict = json.loads(item_str)
                    if item_dict["判断"] == True:
                        true += 1
                    else:
                        false += 1
                except json.JSONDecodeError as e:
                    # 如果某行JSON格式不正确，这里可以打印错误并跳过
                    print(f"解析 JSON 时出错: {e}")
                    print(f"出错的行内容: {item_str}")
                    # 根据需求决定如何处理，下面演示跳过
                    continue
        except Exception as e:
            print(f"读取文件 {outpath} 时出错: {e}")
            # 根据需求决定如何处理。这里演示直接返回，或可改成 pass
            return
    # 检查已处理条目数（断点续处理）
    def count_existing_entries(file_path):
        if not os.path.exists(file_path):
            return 0
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)  # 计算文件行数，即已有条目数

    # 从断点处开始
    start_index = count_existing_entries(outpath)
    print(f"Resuming from index: {start_index}")

    with open('data/chinese/简单生成任务.txt', 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]  # 去掉每行末尾的换行符

    # 跳过已处理的部分
    lines = lines[start_index:]
    
    for sentence in tqdm(lines):
        # print("替换前句子："+sentence)
        sample = {}
        assistant_input = replace_number(i, sentence)
        assistant_output = LLM(assistant_input)
        # print(judge(assistant_input, assistant_output))
        sample["问题"] = assistant_input
        sample["回答"] = assistant_output
        chinese_characters = re.findall(r'[\u4e00-\u9fff]', assistant_output)
        chinese_count = len(chinese_characters)
        sample["模型生成个数"] = chinese_count
        sample["判断"] = judge(assistant_input, assistant_output)

        with open(outpath, 'a', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')  # 添加换行符，以便下一次写入时不会直接接在前一行的末尾
            
        total += 1
        if sample["判断"]== True:
            true += 1
        else:
            false += 1

    item={}
    item[f"生成数字{i}个数成功率"] = true/total
    item[f"生成数字{i}个数失败率"] = false/total
    with open(path, 'a', encoding='utf-8') as f:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')  # 添加换行符，以便下一次写入时不会直接接在前一行的末尾
        
if __name__ == '__main__':
    list = range(1, 101)
    for i in list:
        test(i)    #素数, 奇数, 偶数, 2的幂, 非2的幂, 完全平方数, 阶乘, 质因数