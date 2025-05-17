import os
import re
import json
from datetime import datetime
from tqdm import tqdm
import sys
import importlib.util

# 导入不同的改进策略
def import_strategy(strategy_name):
    try:
        spec = importlib.util.spec_from_file_location(
            strategy_name, 
            f"{os.path.dirname(__file__)}/{strategy_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"导入策略 {strategy_name} 失败: {e}")
        return None

def replace_number(i, sentence, is_chinese=True):
    numbers = re.findall(r'\d+', sentence)
    numbers = [int(num) for num in numbers]
    if i:
        for num in numbers:
            if i:
                random_number = i
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

def test_strategy(strategy_name, i, is_chinese=True, data_dir=None, output_dir=None, data_type=None):
    # 设置默认值
    if data_dir is None:
        if data_type == 'range':
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/range_Q')
        else:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/chinese' if is_chinese else 'data/english')
            
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'result', '中文' if is_chinese else '英文')
    
    # 构建文件路径
    path = os.path.join(output_dir, '得分', f'"{strategy_name}"-数字{i}测试结果.json')
    outpath = os.path.join(output_dir, '输出', f'"{strategy_name}"-数字{i}模型输出结果.json')
    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    # 导入策略模块
    strategy_module = import_strategy(strategy_name)
    if not strategy_module:
        print(f"无法导入策略 {strategy_name}")
        return
    
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
    if data_type == 'range':
        task_file = os.path.join(data_dir, 'CH.txt' if is_chinese else 'EN.txt')
    else:
        task_file = os.path.join(data_dir, '简单生成任务.txt' if is_chinese else 'generate_Q.txt')
        
    try:
        with open(task_file, 'r', encoding='utf-8') as file:
            if data_type == 'range':
                # 处理range_Q格式的数据
                lines = []
                for line in file:
                    try:
                        data = json.loads(line.strip())
                        if isinstance(data, dict):
                            lines.append(data.get('question', ''))
                        else:
                            lines.append(str(data))
                    except:
                        continue
            else:
                # 处理普通格式的数据
                lines = [line.strip() for line in file]
    except Exception as e:
        print(f"读取任务文件出错: {e}")
        return
    
    # 从断点处继续处理
    start_index = sum(1 for _ in open(outpath, 'r', encoding='utf-8')) if os.path.exists(outpath) else 0
    lines = lines[start_index:]
    
    # 处理任务
    for sentence in tqdm(lines, desc=f"测试策略 {strategy_name} - 数字 {i}"):
        assistant_input = replace_number(i, sentence, is_chinese)
        if not assistant_input:
            continue
            
        # 调用策略模块的生成函数
        try:
            if hasattr(strategy_module, 'generate_sentence'):
                assistant_output, _ = strategy_module.generate_sentence(assistant_input)
            else:
                print(f"策略 {strategy_name} 没有 generate_sentence 函数")
                continue
        except Exception as e:
            print(f"生成过程出错: {e}")
            continue
        
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
            f"生成数字{i}个数成功率": true / total,
            f"生成数字{i}个数失败率": false / total
        }
        try:
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"写入统计结果时出错: {e}")

def compare_strategies(strategies, number_range, is_chinese=True, data_type=None):
    results = {}
    for strategy in strategies:
        print(f"\n开始测试策略: {strategy}")
        strategy_results = []
        for i in number_range:
            test_strategy(strategy, i, is_chinese, data_type=data_type)
            # 读取结果
            output_dir = os.path.join(os.path.dirname(__file__), 'result', '中文' if is_chinese else '英文')
            path = os.path.join(output_dir, '得分', f'"{strategy}"-数字{i}测试结果.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            result = json.loads(line.strip())
                            strategy_results.append(result)
                        except:
                            continue
        results[strategy] = strategy_results
    
    # 保存比较结果
    comparison_path = os.path.join(os.path.dirname(__file__), 'result', f'策略比较_{data_type}.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # 要测试的策略列表
    strategies = [
        'step_by_step',
        'condidate_probability',
        'llm_length_improver'
    ]
    
    number_range = range(1, 101)
    
    # 测试普通数据集
    print("开始测试普通数据集...")
    # 测试中文
    print("\n开始中文测试...")
    compare_strategies(strategies, number_range, is_chinese=True, data_type='normal')
    
    # 测试英文
    print("\n开始英文测试...")
    compare_strategies(strategies, number_range, is_chinese=False, data_type='normal')
    
    # 测试range数据集
    print("\n开始测试range数据集...")
    # 测试中文
    print("\n开始中文测试...")
    compare_strategies(strategies, number_range, is_chinese=True, data_type='range')
    
    # 测试英文
    print("\n开始英文测试...")
    compare_strategies(strategies, number_range, is_chinese=False, data_type='range') 