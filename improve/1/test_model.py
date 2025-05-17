import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
import os
from tqdm import tqdm
import re

def load_model(model_name, is_fine_tuned=False):
    """加载模型"""
    print(f"正在加载{'微调后的' if is_fine_tuned else ''}模型: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return model, tokenizer
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise

def generate_text(model, tokenizer, prompt, max_length=2048):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = inputs.to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def count_characters(text, is_chinese=True):
    """计算字符数量"""
    if is_chinese:
        return len(re.findall(r'[\u4e00-\u9fff]', text))
    else:
        return len(re.findall(r'\b\w+\b', text))

def load_test_data(file_path, is_chinese=True):
    """加载测试数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # 提取所有包含数字的指令
        test_data = []
        for line in lines:
            # 从指令中提取数字
            numbers = re.findall(r'\d+', line)
            if numbers:
                test_data.append({
                    'instruction': line,
                    'target_number': int(numbers[0])
                })
        
        return test_data
    except Exception as e:
        print(f"读取测试数据文件出错: {e}")
        return []

def test_model(model, tokenizer, test_data, is_chinese=True):
    """测试模型生成指定数量的字符"""
    results = []
    
    for item in tqdm(test_data, desc="测试中"):
        # 构建prompt
        prompt = f"Instruction: {item['instruction']}\nOutput: "
        
        # 生成文本
        generated_text = generate_text(model, tokenizer, prompt)
        
        # 提取生成的输出（去除prompt部分）
        output = generated_text.split("Output: ")[-1].strip()
        
        # 计算字符数量
        char_count = count_characters(output, is_chinese)
        
        results.append({
            "instruction": item['instruction'],
            "generated_text": output,
            "target_count": item['target_number'],
            "actual_count": char_count,
            "is_correct": char_count == item['target_number']
        })
    
    return results

def save_results(results, model_name, is_fine_tuned=False):
    """保存测试结果"""
    output_dir = "results/model_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    detail_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_{'fine_tuned' if is_fine_tuned else 'base'}_details.json")
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算并保存统计信息
    stats = {
        "total_tests": len(results),
        "correct_count": sum(1 for r in results if r["is_correct"]),
        "accuracy": sum(1 for r in results if r["is_correct"]) / len(results) if results else 0
    }
    
    stats_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_{'fine_tuned' if is_fine_tuned else 'base'}_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试结果已保存到: {output_dir}")
    print(f"准确率: {stats['accuracy']:.2%}")

def main():
    # 配置参数
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    fine_tuned_model_path = "models/fine_tuned"
    chinese_test_path = '../../data/chinese/简单生成任务.txt'
    english_test_path = '../../data/english/generate_Q.txt'
    
    # 加载测试数据
    chinese_test_data = load_test_data(chinese_test_path, is_chinese=True)
    english_test_data = load_test_data(english_test_path, is_chinese=False) if os.path.exists(english_test_path) else []
    
    if not chinese_test_data and not english_test_data:
        print("没有找到有效的测试数据！")
        return
    
    # 测试基础模型
    print("\n测试基础模型...")
    base_model, base_tokenizer = load_model(model_name)
    
    base_results = []
    
    # 测试中文数据
    if chinese_test_data:
        print("\n测试中文数据...")
        results = test_model(base_model, base_tokenizer, chinese_test_data, is_chinese=True)
        base_results.extend(results)
    
    # 测试英文数据
    if english_test_data:
        print("\n测试英文数据...")
        results = test_model(base_model, base_tokenizer, english_test_data, is_chinese=False)
        base_results.extend(results)
    
    save_results(base_results, model_name, is_fine_tuned=False)
    
    # 测试微调后的模型
    print("\n测试微调后的模型...")
    try:
        fine_tuned_model, fine_tuned_tokenizer = load_model(fine_tuned_model_path, is_fine_tuned=True)
        
        fine_tuned_results = []
        
        # 测试中文数据
        if chinese_test_data:
            print("\n测试中文数据...")
            results = test_model(fine_tuned_model, fine_tuned_tokenizer, chinese_test_data, is_chinese=True)
            fine_tuned_results.extend(results)
        
        # 测试英文数据
        if english_test_data:
            print("\n测试英文数据...")
            results = test_model(fine_tuned_model, fine_tuned_tokenizer, english_test_data, is_chinese=False)
            fine_tuned_results.extend(results)
        
        save_results(fine_tuned_results, model_name, is_fine_tuned=True)
        
    except Exception as e:
        print(f"测试微调后的模型时出错: {str(e)}")
        print("请确保已经完成模型微调并保存了微调后的模型。")

if __name__ == "__main__":
    main() 