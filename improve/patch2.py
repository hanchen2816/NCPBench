import re
import random
import json  
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
import os
# 中文
def judge_CN(question, assistant_output):
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

# 英文
def judge_EN(question, assistant_output):
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

# 返回句子中的数字
def return_n(sentence):
    numbers = re.findall(r'\d+', sentence)  # 提取句子中的所有数字
    numbers = [int(num) for num in numbers]  # 转换为数字列表
    # 如果文件中的素数列表有效，进行替换
    for num in numbers:
        if num:
            return numbers[0]
        else:
            return 0

def modify_sentence(assistant_input,result):
    if judge_CN(assistant_input,result) ==False:
        characters = re.findall(r'[\u4e00-\u9fff]', result)# 中文
        # english_words = re.findall(r'\b\w+\b', assistant_output)  # 使用正则表达式找到所有单词
        # english_word_count = len(english_words)  # 统计单词数量
        count = return_n(assistant_input)-len(characters)
        if count<0:
            print(f"需要删除{-1*count}个字")
        elif count>0:
            print(f"需要添加{count}个字")
    else:
        print("大模型生成正确")

if __name__ == '__main__':
    assistant_input = "请用4个字描述你最喜欢的城市风景。"
    result = "江南水乡古镇韵。"
    print("生成的句子：", result)
    modify_sentence(assistant_input,result)