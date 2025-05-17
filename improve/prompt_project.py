import os
import re
import time
import json
from openai import OpenAI
from tqdm import tqdm

def get_response(messages):
    client = OpenAI(
        api_key="sk-4b2369448874441b8efe0befe39b9ba2",
        # api_key="ssk-proj-DiFjog7zlIbXFp5iwDdXbiQtsQQYuNP6Ua8OjSFbHzwn0Fac05AIdrSKD3g_M95V8cUlYlwXA7T3BlbkFJ9LFoTWzNFCgAKnwdmID1swxKhPJIBK19-k6xBCdy3XWG2JiNmNpwywGsF3QhK71vY9rgYTjLoA",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model_name = "qwen2-72b-instruct"  # 定义模型名称
    
    max_retries = 3  # 最大重试次数
    retry_delay = 5  # 重试延迟时间（秒）
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.8,
                top_p=0.8
            )
            return completion.choices[0].message.content, model_name
        except Exception as e:
            print(f"API调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
            if "429" in str(e):  # 如果是速率限制错误
                print(f"遇到速率限制，等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 每次重试后延迟时间翻倍
            else:
                time.sleep(2)  # 其他错误等待2秒
            if attempt == max_retries - 1:  # 最后一次尝试
                print("达到最大重试次数，返回空字符串")
                return "", model_name

def generate_sentence(prompt, assistant_input):
    # 提取目标字数
    target_length = int(re.findall(r'\d+', assistant_input)[0])
    messages = [
        {'role': 'system', 'content': "你是一个智能助手，能够根据用户的需求生成内容，请严格遵守用户的要求，包括内容数量、质量、格式等规则，如果出现数字，要完全按照数字字数生成，不要多也不要少一个字。输出格式为一句话。"},
        {'role': 'user', 'content': "请根据下面要求，生成对应的文段，要求其中除去符号后，汉字的个数必须跟要求的一致，只需要输出一句话，不要输出其他内容:"+str(assistant_input)+f"例如：{str(prompt)}"}
    ]
    
    # 获取模型输出
    response, model_name = get_response(messages)
    
    # 如果响应为空，重试一次
    if not response:
        print("模型返回空响应，重试一次...")
        response, model_name = get_response(messages)
        if not response:
            print("重试后仍然返回空响应，返回空字符串")
            return "", model_name
            
    return response, model_name

def get_last_completed_number(output_dir: str, model_name: str) -> int:
    """
    获取上次完成的数字
    通过检查输出文件中的数据条数来确定是否完成了某个数字的处理
    返回最后完成的数字，如果没有完成的数字则返回0
    """
    completed = 0
    for i in range(1, 21):
        # 检查当前数字的输出文件是否存在，文件名包含模型名称
        output_file = os.path.join(output_dir, f"{model_name}_output_{i}.json")
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"数字 {i} 的文件中包含 {len(data)} 条数据")
                        if len(data) >= 1000:  # 如果数据条数达到1000
                            completed = i
                            continue
                        else:
                            break  # 如果数据不足1000条，从这个数字继续处理
                    else:
                        print(f"数字 {i} 的文件格式不正确")
                        break
            except Exception as e:
                print(f"检查文件 {output_file} 时出错: {e}")
                break
        else:
            print(f"数字 {i} 的文件不存在")
            break
    
    return completed

def replace(sentence, number):
    numbers = re.findall(r'\d+', sentence)  # 提取句子中的所有数字
    numbers = [int(num) for num in numbers]  # 转换为数字列表
    
    for num in numbers:
        # 替换句子中的数字
        sentence = re.sub(str(num), str(number), sentence, 1)  # 只替换第一个匹配的数字

    print("替换后的句子：", sentence)
    return sentence

def save_to_json(file_name, assistant_input, step_results, final_sentence, char_count, is_correct):
    new_data = {
        "input_requirement": assistant_input,
        "steps": step_results,
        "final_result": {
            "generated_sentence": final_sentence,
            "character_count": char_count,
            "is_correct": is_correct
        }
    }
    # 检查文件是否存在
    if os.path.exists(file_name):
        # 如果文件存在，读取现有数据
        with open(file_name, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []  # 如果文件格式错误，初始化为空列表
            except json.JSONDecodeError:
                existing_data = []  # 文件内容无效时初始化为空列表
    else:
        existing_data = []

    # 将新数据追加到现有数据
    existing_data.append(new_data)

    # 将完整数据写回文件
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到 {file_name}")

if __name__ == '__main__':
    # 确保输出目录存在
    output_dir = "improve/result/prompt_project"
    stats_dir = "improve/result/prompt_project/stats"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # 获取上次完成的数字
    model_name = "qwen2-72b-instruct"  # 使用当前实际使用的模型名称
    last_completed = get_last_completed_number(output_dir, model_name)
    print(f"上次完成到数字: {last_completed}")
    for i in range(last_completed + 1, 21):
        with open('data/chinese/简单生成任务.txt', 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
            total_lines = len(lines)
            print(f"总数据条数: {total_lines}")
        
        # 从上次完成的数字的下一个开始

            # 初始化统计数据
            total = 0
            true = 0
            false = 0
            
            # 检查当前数字的输出文件
            output_file = os.path.join(output_dir, f"{model_name}_output_{i}.json")
            start_idx = 0
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            start_idx = len(data)
                            print(f"数字 {i} 的文件中已有 {start_idx} 条数据")
                except Exception as e:
                    print(f"读取文件 {output_file} 时出错: {e}")
                    start_idx = 0
            
            if start_idx >= 1000:
                print(f"数字 {i} 已处理完成，跳过")
                continue
                
            end_idx = min(start_idx + 1000, total_lines)
            current_lines = lines[start_idx:end_idx]
            
            print(f"\n开始处理数字 {i}，处理第 {start_idx+1} 到 {end_idx} 条数据")
            
            for sentence in tqdm(current_lines, desc=f"Processing sentences for number {i}", unit="sentence"):
                try:
                    # 检查句子是否为空
                    if not sentence.strip():
                        print(f"警告：跳过空句子")
                        continue
                        
                    assistant_input = replace(sentence, i)
                    
                    # 检查替换后的句子是否包含数字
                    numbers = re.findall(r'\d+', assistant_input)
                    if not numbers:
                        print(f"警告：句子 '{sentence}' 替换后没有找到数字，跳过处理")
                        continue
                        
                    target_length = int(numbers[0])

                    # 指定个数的例子提示
                    with open("data/prompt/number_case.txt",'r') as file:
                        lines = [line.strip() for line in file]
                        prompt = lines[target_length-1]

                    result, model_name = generate_sentence(prompt, assistant_input)
                    char_count = len(re.findall(r'[\u4e00-\u9fff]', result))
                    is_correct = char_count == target_length

                    print("生成的句子：", result)
                    print("生成的字数：", char_count)
                    print("字数是否正确：", is_correct)

                    # 更新统计数据
                    total += 1
                    if is_correct:
                        true += 1
                    else:
                        false += 1

                    # 保存结果
                    save_to_json(
                        output_file,
                        assistant_input,
                        [],  # 空列表，因为prompt_project没有步骤记录
                        result,
                        char_count,
                        is_correct
                    )
                except Exception as e:
                    print(f"处理句子时出错: {e}")
                    print(f"问题句子: {sentence}")
                    continue

            # 计算并保存统计结果
            stats = {
                "total": total,
                "correct": true,
                "incorrect": false,
                "accuracy": true / total if total > 0 else 0,
                "error_rate": false / total if total > 0 else 0
            }
            
            stats_file = os.path.join(stats_dir, f"{model_name}_stats_{i}.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=4)
            
            print(f"\n数字 {i} 的统计结果：")
            print(f"总数：{total}")
            print(f"正确数：{true}")
            print(f"错误数：{false}")
            print(f"正确率：{stats['accuracy']:.2%}")
            print(f"错误率：{stats['error_rate']:.2%}")
