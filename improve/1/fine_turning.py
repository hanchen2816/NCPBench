import json
import os
from typing import List, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GenerationConfig
)
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm
from datetime import datetime

def load_training_data(data_file: str) -> List[Dict]:
    """加载训练数据"""
    print(f"正在加载训练数据: {data_file}")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print(f"警告: {data_file} 是空文件")
                return []
            data = json.loads(content)
        print(f"成功加载 {len(data)} 条训练数据")
        return data
    except json.JSONDecodeError as e:
        print(f"警告: {data_file} 不是有效的 JSON 文件: {str(e)}")
        return []
    except Exception as e:
        print(f"警告: 加载 {data_file} 时出错: {str(e)}")
        return []

def prepare_dataset(data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 2048):
    """准备数据集"""
    print("正在准备数据集...")
    
    def format_prompt(example: Dict) -> str:
        """格式化提示词"""
        instruction = example["instruction"]
        chain_of_thought = example["chain_of_thought"]
        negative_examples = example["negative_examples"]
        
        # 构建思维链文本
        cot_text = "思维链：\n"
        for step in chain_of_thought:
            cot_text += f"- {step['step']}: {step['action']}\n"
        
        # 构建错误示例文本
        neg_text = "\n错误示例：\n"
        for neg in negative_examples:
            neg_text += f"输出: {neg['output']}\n"
            neg_text += f"错误分析: {neg['error_analysis']['error_type']}\n"
            neg_text += f"修复策略: {neg['error_analysis']['repair_strategy']}\n"
        
        # 组合完整提示词
        formatted_text = f"指令：{instruction}\n\n{cot_text}{neg_text}"
        return formatted_text.strip()
    
    # 格式化数据
    formatted_data = []
    for example in tqdm(data, desc="格式化数据"):
        formatted_text = format_prompt(example)
        formatted_data.append({"text": formatted_text})
    
    # 创建数据集
    dataset = Dataset.from_list(formatted_data)
    
    # 对数据集进行分词
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_model(
    model_name: str,
    train_data: List[Dict],
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 100,
    max_length: int = 2048,
    use_wandb: bool = True,
    offline: bool = False
):
    """训练模型"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化wandb
    if use_wandb:
        try:
            wandb.init(
                project="deepseek-finetuning",
                name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model_name": model_name,
                    "num_train_epochs": num_train_epochs,
                    "batch_size": per_device_train_batch_size,
                    "learning_rate": learning_rate,
                    "max_length": max_length
                },
                mode="offline" if offline else "online"
            )
        except Exception as e:
            print(f"警告: wandb初始化失败: {str(e)}")
            print("继续训练，但不使用wandb进行监控...")
            use_wandb = False
    
    # 加载模型和分词器
    print("正在加载模型和分词器...")
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
    
    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # 准备数据集
    print("正在准备数据集...")
    tokenized_dataset = prepare_dataset(train_data, tokenizer, max_length)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=True,
        report_to="wandb" if use_wandb else "none"
    )
    
    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if use_wandb:
        wandb.finish()
    
    print(f"训练完成！模型已保存到: {output_dir}")
    return trainer

def evaluate_model(trainer: Trainer, test_data: List[Dict]):
    """评估模型"""
    print("开始评估模型...")
    
    # 准备测试数据
    test_dataset = prepare_dataset(test_data, trainer.tokenizer)
    
    # 进行评估
    eval_results = trainer.evaluate(test_dataset)
    print(f"评估结果: {eval_results}")
    
    return eval_results

def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1
) -> List[str]:
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = inputs.to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    generated_texts = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 配置参数
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    chinese_data_file = "data/CoT/chinese_dataset.json"
    english_data_file = "data/CoT/english_dataset.json"
    output_dir = "models/fine_tuned"
    
    try:
        # 加载训练数据
        print("\n加载中文训练数据...")
        chinese_data = load_training_data(chinese_data_file)
        
        print("\n加载英文训练数据...")
        english_data = load_training_data(english_data_file)
        
        # 合并数据集
        all_data = chinese_data + english_data
        if not all_data:
            raise ValueError("没有可用的训练数据！请确保至少有一个数据集文件包含有效数据。")
            
        print(f"\n总训练数据量: {len(all_data)} 条")
        print(f"中文数据量: {len(chinese_data)} 条")
        print(f"英文数据量: {len(english_data)} 条")
        
        # 训练模型
        trainer = train_model(
            model_name=model_name,
            train_data=all_data,
            output_dir=output_dir,
            use_wandb=True,  # 是否使用wandb进行训练监控
            offline=True     # 使用离线模式
        )
        
        # 评估模型
        eval_results = evaluate_model(trainer, all_data)
        
        # 测试生成
        print("\n测试文本生成...")
        test_prompts = [
            "请生成一个包含5个字的句子，描述春天的景色。",
            "Generate a 5-word sentence about spring."
        ]
        
        for prompt in test_prompts:
            print(f"\n输入: {prompt}")
            generated_texts = generate_text(
                trainer.model,
                trainer.tokenizer,
                prompt,
                max_length=2048
            )
            print(f"输出: {generated_texts[0]}")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 