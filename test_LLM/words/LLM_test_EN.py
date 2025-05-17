import re
import random
import json  
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
import os
# 获取今天的日期
today = datetime.now()
# 格式化输出为 "月.日" 形式
formatted_date = today.strftime("%m.%-d").lstrip("0")

def replace(number_list, sentence):
    numbers = re.findall(r'\d+', sentence)  # 提取句子中的所有数字
    numbers = [int(num) for num in numbers]  # 转换为数字列表
    # 如果文件中的素数列表有效，进行替换
    if number_list:
        for num in numbers:
            # 确保number_list不为空
            if number_list:
                random_number = random.choice(number_list)  # 随机选择一个素数
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
        # 填写DashScope服务的base_url
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0.8,
        top_p=0.8
    )
    return completion
  
def LLM(assistant_input):
    messages = [
        {'role': 'system', 'content': "You are an intelligent assistant that can generate content according to the needs of users, please strictly follow the user's requirements, including rules such as the quantity, quality, and format of content, and if there are numbers, they should be generated exactly according to the number of words, not more and not less than one word. The output format is one sentence."},
        {'role': 'user', 'content': "Please generate the corresponding paragraph according to the following requirements, and the number of words after the place symbol must be consistent with the requirements."+str(assistant_input)}
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
    # 统计英文单词数
    english_words = re.findall(r'\b\w+\b', assistant_output)  # 使用正则表达式找到所有单词
    english_word_count = len(english_words)  # 统计单词数量
    print(f"大模型生成的句子：{assistant_output}")
    print("大模型生成的字的个数："+ str(english_word_count))
    if english_word_count == numbers[0]:
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

def test(type):
    path = f'result/words/英文/得分/"Gpt-4omini"-{type}测试结果.json'
    outpath = f'result/words/英文/输出/"Gpt-4omini"-{type}模型输出结果.json'

    # 如果需要确保目录存在，可以在这里检查并创建
    dir_path = os.path.dirname(outpath)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    lines = []
    total = 0
    true = 0
    false = 0
    
    # 如果 outpath 文件已存在，则尝试读取之前的内容
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

    # 函数：统计已处理条目数（断点续处理）
    def count_existing_entries(file_path):
        if not os.path.exists(file_path):
            return 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)  # 计算文件行数
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return 0

    # 从断点处开始
    start_index = count_existing_entries(outpath)
    print(f"Resuming from index: {start_index}")

    # 读取生成任务
    try:
        with open('data/english/generate_Q.txt', 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
    except FileNotFoundError:
        print("找不到文件，无法继续。")
        return
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # 跳过已处理的部分
    lines = lines[start_index:]

    # 开始处理行
    for sentence in tqdm(lines):
        sample = {}
        # 生成数字列表
        number_list = number_lists(type)
        
        # 替换数字
        assistant_input = replace(number_list, sentence)

        # 调用模型
        assistant_output = LLM(assistant_input)

        # 组装结果
        sample["问题"] = assistant_input
        sample["回答"] = assistant_output
        # 统计英文单词数
        english_words = re.findall(r'\b\w+\b', assistant_output)
        english_word_count = len(english_words)
        sample["英文单词个数"] = english_word_count
        sample["判断"] = judge(assistant_input, assistant_output)

        # 写入文件 (注意要确保目录存在)
        try:
            with open(outpath, 'a', encoding='utf-8') as f:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"写入 {outpath} 时出错: {e}")
            # 根据需求决定是否终止、重试或跳过
            continue
        
        total += 1
        if sample["判断"] == True:
            true += 1
        else:
            false += 1

    # 计算最终成功率和失败率
    item = {}
    if total > 0:
        item[f"生成{type}个数成功率"] = true / total
        item[f"生成{type}个数失败率"] = false / total
    else:
        item[f"生成{type}个数成功率"] = 0
        item[f"生成{type}个数失败率"] = 0

    # 写入统计结果
    stats_dir = os.path.dirname(path)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir, exist_ok=True)

    try:
        with open(path, 'a', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f"写入统计结果 {path} 时出错: {e}")
        
if __name__ == '__main__':
    list = ["素数", "奇数", "偶数", "2的幂", "非2的幂", "完全平方数", "阶乘", "质因数"]
    for i in list:
        test(i)    #素数, 奇数, 偶数, 2的幂, 非2的幂, 完全平方数, 阶乘, 质因数