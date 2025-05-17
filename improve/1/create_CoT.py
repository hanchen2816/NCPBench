import json
import os
from typing import List, Dict, Any
import random
import re
import time
from openai import OpenAI

def get_response(messages, max_retries=10, retry_delay=5):
    """调用大模型API，添加重试机制"""
    client = OpenAI(
        api_key="40d7da94-50c3-48b3-9856-9cb2e315a215",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    model_name = "deepseek-r1-250120"
    
    for attempt in range(max_retries):
        try:
            print(f"尝试调用API (第{attempt + 1}次)...")
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.8,
                top_p=0.8
            )
            response = completion.choices[0].message.content
            if not response:
                raise Exception("API返回空响应")
            print("API调用成功")
            return response, model_name
        except Exception as e:
            print(f"API调用失败 (第{attempt + 1}次): {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
            else:
                print("达到最大重试次数，返回空响应")
                return "", model_name

def replace_number_in_task(task: str) -> str:
    """替换任务中的数字为1-20之间的随机数"""
    def replace_number(match):
        return str(random.randint(1, 20))
    
    # 使用正则表达式替换所有数字
    return re.sub(r'\d+', replace_number, task)

def load_tasks(task_file: str) -> List[str]:
    """加载任务文件并随机替换数字"""
    with open(task_file, 'r', encoding='utf-8') as f:
        tasks = [line.strip() for line in f if line.strip()]
        # 对每个任务进行数字替换
        return [replace_number_in_task(task) for task in tasks]

def generate_chain_of_thought(instruction: str) -> List[Dict[str, Any]]:
    """使用大模型生成思维链"""
    # 解析指令中的数字约束
    number_match = re.search(r'(\d+)', instruction)
    if not number_match:
        print("警告：无法从指令中提取数字约束")
        return []
    
    target_number = int(number_match.group(1))
    print(f"目标字数：{target_number}")
    
    # 构建提示词
    prompt = f"""请为以下指令生成详细的思维链步骤：
指令：{instruction}

要求：
1. 每个步骤都要包含具体的分析和计算
2. 需要包含字数统计和验证
3. 给出具体的示例

请按照以下格式输出：
1. 理解约束：分析数字约束
2. 内容规划：分析可用字数
3. 生成策略：设计句子结构
4. 自我验证：验证字数

示例输出：
1. 理解约束：分析数字'5'为严格上限，需要生成不超过5个字的句子
2. 内容规划：选择核心词"春天"(2字)，剩余可用3字
3. 生成策略：采用[2+3]结构，使用"春日"作为开头，搭配动词短语
4. 自我验证：逐字计数：春(1)/日(2)/暖(3)/风(4)/吹(5)，总字数=5，符合要求

JSON格式示例：
{{
    "chain_of_thought": [
        {{
            "step": "理解约束",
            "action": "分析数字'5'为严格上限，需要生成不超过5个字的句子",
            "checkpoint": "字数≤5"
        }},
        {{
            "step": "内容规划",
            "action": "选择核心词'春天'(2字)，剩余可用3字",
            "variables": {{"used": 2, "remaining": 3}}
        }},
        {{
            "step": "生成策略",
            "action": "采用[2+3]结构，使用'春日'作为开头，搭配动词短语",
            "example": "春日暖风吹"
        }},
        {{
            "step": "自我验证",
            "action": "逐字计数：春(1)/日(2)/暖(3)/风(4)/吹(5)，总字数=5，符合要求",
            "result": {{"actual_length": 5}}
        }}
    ]
}}
"""
    
    messages = [
        {'role': 'system', 'content': "你是一个专业的文本生成助手，擅长分析文本约束和生成策略。"},
        {'role': 'user', 'content': prompt}
    ]
    
    response, _ = get_response(messages)
    if not response:
        print("警告：思维链生成失败，API返回空响应")
        return []
    
    print("成功获取思维链响应")
    print("\nAPI返回的响应内容：")
    print(response)
    print("\n开始解析响应...")
    
    # 尝试直接解析JSON
    try:
        # 查找JSON部分
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            print("\n找到JSON部分：")
            print(json_str)
            data = json.loads(json_str)
            if 'chain_of_thought' in data:
                steps = data['chain_of_thought']
                print(f"\n成功从JSON中解析出 {len(steps)} 个思维链步骤")
                return steps
    except Exception as e:
        print(f"\nJSON解析失败: {str(e)}")
    
    # 如果JSON解析失败，尝试按行解析
    print("\n尝试按行解析...")
    steps = []
    current_step = None
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        print(f"处理行: {line}")
        
        if line.startswith('1. 理解约束'):
            current_step = {
                "step": "理解约束",
                "action": line.split('：')[1].strip(),
                "checkpoint": f"字数≤{target_number}"
            }
        elif line.startswith('2. 内容规划'):
            current_step = {
                "step": "内容规划",
                "action": line.split('：')[1].strip(),
                "variables": {"target_length": target_number}
            }
        elif line.startswith('3. 生成策略'):
            current_step = {
                "step": "生成策略",
                "action": line.split('：')[1].strip(),
                "example": "示例句子"  # 这里可以进一步提取示例
            }
        elif line.startswith('4. 自我验证'):
            current_step = {
                "step": "自我验证",
                "action": line.split('：')[1].strip(),
                "result": {"actual_length": target_number}
            }
        
        if current_step:
            steps.append(current_step)
            print(f"添加步骤: {current_step}")
    
    if not steps:
        print("警告：无法从响应中解析出思维链步骤")
        return []
        
    print(f"成功解析出 {len(steps)} 个思维链步骤")
    return steps

def generate_negative_example(instruction: str) -> Dict[str, Any]:
    """使用大模型生成负面示例"""
    # 解析指令中的数字约束
    number_match = re.search(r'(\d+)', instruction)
    if not number_match:
        print("警告：无法从指令中提取数字约束")
        return {}
    
    target_number = int(number_match.group(1))
    print(f"目标字数：{target_number}")
    
    # 构建提示词
    prompt = f"""请为以下指令生成一个违反字数限制的示例，并分析错误原因：
指令：{instruction}

要求：
1. 生成一个超出字数限制的示例
2. 分析错误原因
3. 提供修复策略

请按照以下格式输出：
示例：[生成的内容]
错误分析：
- 超出字数：X
- 错误类型：[具体错误类型]
- 修复策略：[具体修复方法]

示例输出：
示例：春日暖风轻拂
错误分析：
- 超出字数：6字（原要求5字）
- 错误类型：动词短语冗余，使用了"轻拂"而不是"拂"
- 修复策略：将"轻拂"替换为"拂"，得到"春日暖风吹"

JSON格式示例：
{{
    "output": "春日暖风轻拂",
    "error_analysis": {{
        "over_length": 6,
        "error_type": "动词短语冗余，使用了'轻拂'而不是'拂'",
        "repair_strategy": "将'轻拂'替换为'拂'，得到'春日暖风吹'"
    }}
}}
"""
    
    messages = [
        {'role': 'system', 'content': "你是一个专业的文本分析助手，擅长识别文本错误并提供修复建议。"},
        {'role': 'user', 'content': prompt}
    ]
    
    response, _ = get_response(messages)
    if not response:
        print("警告：负面示例生成失败，API返回空响应")
        return {}
    
    print("成功获取负面示例响应")
    print("\nAPI返回的响应内容：")
    print(response)
    print("\n开始解析响应...")
    
    # 尝试直接解析JSON
    try:
        # 查找JSON部分
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            print("\n找到JSON部分：")
            print(json_str)
            data = json.loads(json_str)
            if 'output' in data and 'error_analysis' in data:
                print("\n成功从JSON中解析出负面示例")
                return data
    except Exception as e:
        print(f"\nJSON解析失败: {str(e)}")
    
    # 如果JSON解析失败，尝试按行解析
    print("\n尝试按行解析...")
    output = ""
    error_analysis = {}
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        print(f"处理行: {line}")
        
        # 尝试多种可能的格式
        if line.startswith('示例：') or line.startswith('示例:') or line.startswith('输出：') or line.startswith('输出:'):
            output = line.split('：')[-1].strip()
            if output.startswith('[') and output.endswith(']'):
                output = output[1:-1]
            print(f"找到示例: {output}")
            
        elif line.startswith('- 超出字数：') or line.startswith('- 超出字数:') or line.startswith('超出字数：') or line.startswith('超出字数:'):
            length_text = line.split('：')[-1].strip()
            print(f"处理超出字数: {length_text}")
            # 使用正则表达式提取数字
            length_match = re.search(r'(\d+)', length_text)
            if length_match:
                error_analysis['over_length'] = int(length_match.group(1))
                print(f"提取到超出字数: {error_analysis['over_length']}")
            else:
                error_analysis['over_length'] = 0
                
        elif line.startswith('- 错误类型：') or line.startswith('- 错误类型:') or line.startswith('错误类型：') or line.startswith('错误类型:'):
            error_analysis['error_type'] = line.split('：')[-1].strip()
            print(f"找到错误类型: {error_analysis['error_type']}")
            
        elif line.startswith('- 修复策略：') or line.startswith('- 修复策略:') or line.startswith('修复策略：') or line.startswith('修复策略:'):
            error_analysis['repair_strategy'] = line.split('：')[-1].strip()
            print(f"找到修复策略: {error_analysis['repair_strategy']}")
    
    # 确保所有必要的字段都存在
    if 'over_length' not in error_analysis:
        error_analysis['over_length'] = 0
    if 'error_type' not in error_analysis:
        error_analysis['error_type'] = "未指定错误类型"
    if 'repair_strategy' not in error_analysis:
        error_analysis['repair_strategy'] = "未提供修复策略"
    
    if not output:
        print("警告：未能从响应中提取出示例")
        return {}
        
    print("成功解析负面示例")
    return {
        "output": output,
        "error_analysis": error_analysis
    }

def save_dataset(dataset: List[Dict], output_file: str):
    """保存数据集"""
    try:
        # 验证数据集
        if not dataset:
            print("警告：数据集为空，不进行保存")
            return
            
        # 验证每个条目
        for i, entry in enumerate(dataset):
            if not isinstance(entry, dict):
                print(f"警告：第{i+1}个条目不是字典类型")
                continue
                
            required_fields = ["instruction", "chain_of_thought", "negative_examples"]
            missing_fields = [field for field in required_fields if field not in entry]
            if missing_fields:
                print(f"警告：第{i+1}个条目缺少必要字段: {missing_fields}")
                continue
                
            if not entry["chain_of_thought"]:
                print(f"警告：第{i+1}个条目的思维链为空")
                continue
                
            if not entry["negative_examples"]:
                print(f"警告：第{i+1}个条目的负面示例为空")
                continue
        
        # 创建目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存数据
        print(f"\n正在保存数据集到 {output_file}")
        print(f"数据集大小: {len(dataset)} 个条目")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        # 验证保存的文件
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"文件已保存，大小: {file_size} 字节")
            
            # 尝试读取保存的文件
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                if len(saved_data) == len(dataset):
                    print("数据验证成功：保存的数据集大小与原始数据集一致")
                else:
                    print(f"警告：保存的数据集大小({len(saved_data)})与原始数据集大小({len(dataset)})不一致")
        else:
            print("错误：文件未能成功保存")
            
    except Exception as e:
        print(f"保存数据集时出错: {str(e)}")
        raise

def convert_to_training_format(entry: Dict, is_english: bool = False) -> Dict:
    """将数据集条目转换为训练格式"""
    # 构建思维链文本
    chain_text = ""
    for step in entry["chain_of_thought"]:
        if is_english:
            chain_text += f"{step['step']}: {step['action']}\n"
        else:
            chain_text += f"{step['step']}：{step['action']}\n"
    
    # 构建负面示例文本
    negative_text = ""
    for example in entry["negative_examples"]:
        if is_english:
            negative_text += f"Error Example: {example['output']}\n"
            negative_text += f"Error Analysis:\n"
            negative_text += f"- Over Length: {example['error_analysis']['over_length']}\n"
            negative_text += f"- Error Type: {example['error_analysis']['error_type']}\n"
            negative_text += f"- Repair Strategy: {example['error_analysis']['repair_strategy']}\n"
        else:
            negative_text += f"错误示例：{example['output']}\n"
            negative_text += f"错误分析：\n"
            negative_text += f"- 超出字数：{example['error_analysis']['over_length']}\n"
            negative_text += f"- 错误类型：{example['error_analysis']['error_type']}\n"
            negative_text += f"- 修复策略：{example['error_analysis']['repair_strategy']}\n"
    
    # 构建训练格式
    if is_english:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional text generation assistant, skilled in analyzing text constraints and generation strategies. Please follow these steps:\n1. Understand Constraints\n2. Content Planning\n3. Generation Strategy\n4. Self-Verification"
                },
                {
                    "role": "user",
                    "content": entry["instruction"]
                },
                {
                    "role": "assistant",
                    "content": f"Let me think about this step by step:\n\n{chain_text}\n\nNegative Example Analysis:\n{negative_text}"
                }
            ]
        }
    else:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的文本生成助手，擅长分析文本约束和生成策略。请按照以下步骤思考：\n1. 理解约束\n2. 内容规划\n3. 生成策略\n4. 自我验证"
                },
                {
                    "role": "user",
                    "content": entry["instruction"]
                },
                {
                    "role": "assistant",
                    "content": f"让我按照步骤来思考这个问题：\n\n{chain_text}\n\n负面示例分析：\n{negative_text}"
                }
            ]
        }

def save_training_dataset(dataset: List[Dict], output_file: str, is_english: bool = False):
    """保存训练格式的数据集"""
    try:
        # 转换为训练格式
        training_data = []
        for entry in dataset:
            training_entry = convert_to_training_format(entry, is_english)
            training_data.append(training_entry)
        
        # 创建目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存数据
        print(f"\n正在保存训练数据集到 {output_file}")
        print(f"数据集大小: {len(training_data)} 个条目")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
            
        # 验证保存的文件
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"文件已保存，大小: {file_size} 字节")
            
            # 尝试读取保存的文件
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                if len(saved_data) == len(training_data):
                    print("数据验证成功：保存的数据集大小与原始数据集一致")
                else:
                    print(f"警告：保存的数据集大小({len(saved_data)})与原始数据集大小({len(training_data)})不一致")
        else:
            print("错误：文件未能成功保存")
            
    except Exception as e:
        print(f"保存训练数据集时出错: {str(e)}")
        raise

def generate_dataset(task_file: str, output_file: str, is_english: bool = False):
    """生成完整的数据集"""
    tasks = load_tasks(task_file)
    dataset = []
    
    print(f"开始处理任务文件: {task_file}")
    print(f"共加载 {len(tasks)} 个任务")
    
    # 检查文件是否存在
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if existing_data:
                    print(f"发现已存在的数据集，包含 {len(existing_data)} 个条目")
                    dataset = existing_data
                    # 从已有数据集的长度开始处理
                    start_index = len(dataset)
                    tasks = tasks[start_index:]
                    print(f"将从第 {start_index + 1} 个任务开始处理")
        except Exception as e:
            print(f"读取已存在的数据集时出错: {str(e)}")
    
    for i, task in enumerate(tasks):
        try:
            print(f"\n处理任务 {i+1}/{len(tasks)}: {task}")
            
            # 生成思维链
            print("正在生成思维链...")
            chain_of_thought = generate_chain_of_thought(task)
            if not chain_of_thought:
                print("警告：思维链生成失败，跳过此任务")
                continue
                
            # 生成负面示例
            print("正在生成负面示例...")
            negative_example = generate_negative_example(task)
            if not negative_example:
                print("警告：负面示例生成失败，跳过此任务")
                continue
        
            entry = {
                "instruction": task,
                "chain_of_thought": chain_of_thought,
                "negative_examples": [negative_example]
            }
            dataset.append(entry)
        
            # 每处理5个任务保存一次
            if (i + 1) % 5 == 0:
                print(f"\n保存当前进度...")
                try:
                    # 保存原始数据集
                    save_dataset(dataset, output_file)
                    # 保存训练格式数据集
                    training_output_file = output_file.replace('.json', '_training.json')
                    save_training_dataset(dataset, training_output_file, is_english)
                    print(f"已保存 {len(dataset)} 个任务到 {output_file} 和 {training_output_file}")
                except Exception as e:
                    print(f"保存进度时出错: {str(e)}")
                
        except Exception as e:
            print(f"处理任务时出错: {str(e)}")
            continue
    
    # 最终保存
    try:
        print(f"\n保存最终结果...")
        # 保存原始数据集
        save_dataset(dataset, output_file)
        # 保存训练格式数据集
        training_output_file = output_file.replace('.json', '_training.json')
        save_training_dataset(dataset, training_output_file, is_english)
        print(f"完成所有任务，共保存 {len(dataset)} 个有效任务到 {output_file} 和 {training_output_file}")
        
    except Exception as e:
        print(f"保存最终结果时出错: {str(e)}")

def main():
    # 设置随机种子以确保结果可复现
    random.seed(42)
    
    try:
        # 生成中文数据集
        print("\n开始生成中文数据集...")
        chinese_task_file = "data/chinese/简单生成任务.txt"
        chinese_output_file = "data/CoT/chinese_dataset.json"
        generate_dataset(chinese_task_file, chinese_output_file, is_english=False)
        
        # 验证中文数据集
        if os.path.exists(chinese_output_file):
            with open(chinese_output_file, 'r', encoding='utf-8') as f:
                chinese_data = json.load(f)
                print(f"中文数据集大小: {len(chinese_data)} 个条目")
        else:
            print("警告：中文数据集文件未生成")
    
        # 生成英文数据集
        print("\n开始生成英文数据集...")
        english_task_file = "data/english/generate_Q.txt"
        english_output_file = "data/CoT/english_dataset.json"
        generate_dataset(english_task_file, english_output_file, is_english=True)
        
        # 验证英文数据集
        if os.path.exists(english_output_file):
            with open(english_output_file, 'r', encoding='utf-8') as f:
                english_data = json.load(f)
                print(f"英文数据集大小: {len(english_data)} 个条目")
        else:
            print("警告：英文数据集文件未生成")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()