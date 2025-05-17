import os
import re
import json
from openai import OpenAI
import random
from tqdm import tqdm
import time
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
                # model="qwen1.5-72b-chat",
                messages=messages,
                temperature=0.8,
                top_p=0.8,
                max_tokens=10  # 增加 token 限制
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

def rollback_last_steps(step_results, generated_text, steps_to_rollback=2):
    """
    回退最后几步生成的结果。
    """
    if step_results:
        for _ in range(steps_to_rollback):
            if step_results:  # 确保还有步骤可以回退
                last_step = step_results.pop()
                generated_text = last_step["current_generated_text"][:-len(last_step["response"])]
            else:
                break
        current_length = len(re.findall(r'[\u4e00-\u9fff]', generated_text))
        print(f"回退成功：当前文本为 '{generated_text}'")
        return generated_text, current_length
    return generated_text, len(re.findall(r'[\u4e00-\u9fff]', generated_text))


def generate_sentence(assistant_input, max_retries=5):
    target_length = int(re.findall(r'\d+', assistant_input)[0])
    generated_text = ""
    current_length = 0
    messages = [
        {'role': 'system', 'content': "你是一个智能助手，能够根据用户的需求生成内容，请严格遵守用户的要求，包括内容数量、质量、格式等规则，如果出现数字，要完全按照数字字数生成，不要多也不要少一个字。输出格式为一句话。"},
        {'role': 'user', 'content': f"请根据下面要求，生成对应的文段，要求其中除去符号后，汉字的个数必须跟要求的一致，只需要输出一句话，不要输出其他内容: {assistant_input}"}
    ]

    print("生成过程开始...")
    retries = 0  # 当前生成的重试计数器
    rollback_counter = 0  # 回退计数器
    step_results = []
    model_name = None  # 初始化模型名称
    max_tokens = 10  # 定义最大token数
    last_valid_text = ""  # 存储最后一次有效的文本

    while current_length < target_length:
        remaining_length = target_length - current_length
        user_prompt = (
            f"你已经写了 {current_length} 个字：{generated_text}，"
            f"现在只能再写 {remaining_length} 个字完成我需要的句子，请生成一个词。"
        )
        messages.append({'role': 'user', 'content': user_prompt})
        print(f"提示模型：{user_prompt}")

        response, model_name = get_response(messages)  # 获取响应和模型名称
        print(f"模型输出：{response}")

        # 如果响应为空，重试一次
        if not response:
            print("模型返回空响应，重试一次...")
            response, model_name = get_response(messages)
            if not response:
                print("重试后仍然返回空响应，跳过当前步骤")
                continue

        if response:
            chinese_characters = re.findall(r'[\u4e00-\u9fff]', response)
            response_length = len(chinese_characters)

            if current_length + response_length <= target_length:
                # 更新生成的文本
                generated_text += response
                current_length += response_length
                retries = 0  # 重置重试计数器
                rollback_counter = 0  # 重置回退计数器
                step_results.append({
                    "step": len(step_results) + 1,
                    "prompt": user_prompt,
                    "response": response,
                    "current_generated_text": generated_text,
                    "current_length": current_length
                })
                last_valid_text = generated_text  # 更新最后一次有效的文本
                print(f"当前生成的文本：{generated_text}（长度：{current_length}）")
            else:
                print(f"生成的词超过目标长度，跳过：{response}")
                # 只有在接近目标长度时才进行回退
                if remaining_length < max_tokens and rollback_counter < 1:
                    print(f"接近目标长度，尝试回退...")
                    generated_text, current_length = rollback_last_steps(step_results, generated_text)
                    rollback_counter += 1
                retries += 1
        else:
            print("生成的词无效，重新请求。")
            retries += 1

        # 超过最大重试次数，执行回退或跳过
        if retries >= max_retries:
            if remaining_length < max_tokens and rollback_counter < 1:  # 只有在接近目标长度时才进行回退
                print(f"重试次数过多，尝试回退...")
                generated_text, current_length = rollback_last_steps(step_results, generated_text)
                rollback_counter += 1
                retries = 0  # 重置重试计数器
            else:
                print(f"重试和回退次数过多，返回最后一次有效的文本：{last_valid_text}")
                return last_valid_text, model_name

        messages.append({'role': 'assistant', 'content': response})

    print("生成过程结束！")
    return generated_text, model_name



    
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

def replace(sentence, number):
    numbers = re.findall(r'\d+', sentence)  # 提取句子中的所有数字
    numbers = [int(num) for num in numbers]  # 转换为数字列表
    
    for num in numbers:
        # 替换句子中的数字
        sentence = re.sub(str(num), str(number), sentence, 1)  # 只替换第一个匹配的数字

    print("替换后的句子：", sentence)
    return sentence

def count_existing_entries(file_path):
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data)  # 返回JSON数组的长度
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0

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

def main():
    # 确保输出目录存在
    output_dir = "improve/result/step_by_step"
    stats_dir = "improve/result/step_by_step/stats"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    model_name = "qwen2-72b-instruct"  # 使用当前实际使用的模型名称
    
    with open('data/chinese/简单生成任务.txt', 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
        total_lines = len(lines)
        print(f"总数据条数: {total_lines}")
        
        # 从1开始处理每个数字
        for i in range(1, 21):
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
            
            # 处理剩余的所有数据
            current_lines = lines[start_idx:]
            
            print(f"\n开始处理数字 {i}，处理第 {start_idx+1} 到 {total_lines} 条数据")
            
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
                    result, model_name = generate_sentence(assistant_input)
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
                        [],  # 空列表，因为step_by_step没有步骤记录
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

if __name__ == '__main__':
    main()

