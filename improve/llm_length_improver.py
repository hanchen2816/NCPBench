import re
from openai import OpenAI
import math
from itertools import combinations
import jieba
import jieba.posseg as pseg
from typing import List, Tuple, Dict
import os
import json
from tqdm import tqdm
import logging
import time

# 确保日志目录存在
log_dir = "improve/result/llm_length_improver/logs"
os.makedirs(log_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'llm_length_improver.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 初始化jieba
logging.info("正在初始化jieba分词器...")
jieba.initialize()
logging.info("jieba分词器初始化完成")

class LLMLengthImprover:
    def __init__(self):
        """
        初始化LLM长度改进器
        """
        logging.info("正在初始化LLM长度改进器...")
        self.client = OpenAI(
            api_key="40d7da94-50c3-48b3-9856-9cb2e315a215",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        self.word_weights = {
            'n': 0.8,    # 名词
            'v': 0.7,    # 动词
            'a': 0.7,    # 形容词
            't': 0.6,    # 时间词
            'r': 0.5,    # 代词
            'd': 0.4,    # 副词
            'p': 0.3,    # 介词
            'c': 0.2,    # 连词
            'u': 0.2,    # 助词
            'x': 0.1     # 标点符号
        }
        logging.info("LLM长度改进器初始化完成")
    
    def get_llm_response(self, prompt: str, target_length: int) -> str:
        """
        获取LLM的响应并调整到目标长度
        
        Args:
            prompt: 输入提示
            target_length: 目标字数
        
        Returns:
            调整后的响应
        """
        messages = [
            {
                'role': 'system',
                'content': "你是一个智能助手，能够根据用户的需求生成内容，请严格遵守用户的要求，包括内容数量、质量、格式等规则，如果出现数字，要完全按照数字字数生成，不要多也不要少一个字。输出格式为一句话。"
            },
            {
                'role': 'user',
                'content': f"请根据下面要求，生成对应的文段，要求其中除去符号后，汉字的个数必须跟要求的一致：{prompt}"
            }
        ]
        
        max_retries = 3
        timeout = 30  # 设置30秒超时
        
        for attempt in range(max_retries):
            try:
                logging.info(f"尝试调用API (第{attempt + 1}次)")
                completion = self.client.chat.completions.create(
                    model="deepseek-r1-250120",
                    messages=messages,
                    temperature=0.8,
                    top_p=0.8,
                    timeout=timeout  # 添加超时设置
                )
                original_response = completion.choices[0].message.content
                logging.info("API调用成功")
                
                # 调整响应长度
                adjusted_response = self._adjust_length(original_response, target_length)
                return adjusted_response
                
            except Exception as e:
                logging.error(f"API调用失败 (第{attempt + 1}次): {str(e)}")
                if attempt < max_retries - 1:
                    logging.info(f"等待5秒后重试...")
                    time.sleep(5)  # 等待5秒后重试
                else:
                    logging.error("达到最大重试次数，返回None")
                    return None
    
    def _adjust_length(self, text: str, target_length: int) -> str:
        """
        调整文本长度到目标长度
        
        Args:
            text: 原始文本
            target_length: 目标长度
        
        Returns:
            调整后的文本
        """
        # 提取候选词
        candidates = self._extract_candidates(text)
        
        # 调整概率
        adjusted_probs = self._adjust_probability(candidates, target_length)
        
        # 生成组合
        tokens = [token for token, _ in candidates]
        valid_combinations = self._generate_combinations(tokens, target_length)
        
        # 计算最佳组合
        best_combination = self._find_best_combination(valid_combinations, adjusted_probs)
        
        return ''.join(best_combination) if best_combination else text
    
    def _extract_candidates(self, text: str) -> List[Tuple[str, float]]:
        """
        从文本中提取候选词及其概率
        """
        words = pseg.cut(text)
        words_with_weights = []
        
        for word, flag in words:
            if word.strip():
                weight = self.word_weights.get(flag[0], 0.5)
                words_with_weights.append((word, weight))
        
        # 计算概率
        total_weight = sum(weight for _, weight in words_with_weights)
        candidates = [(word, weight/total_weight) for word, weight in words_with_weights]
        
        # 添加单字
        single_chars = set()
        for word, _ in candidates:
            if len(word) >= 2:
                for char in word:
                    single_chars.add(char)
        
        # 为单字设置概率
        base_single_char_prob = 0.1
        for char in single_chars:
            candidates.append((char, base_single_char_prob))
        
        # 归一化
        total_prob = sum(prob for _, prob in candidates)
        return [(word, prob/total_prob) for word, prob in candidates]
    
    def _adjust_probability(self, candidates: List[Tuple[str, float]], target_length: int, beta: float = 2.0) -> List[Tuple[str, float]]:
        """
        调整候选词的概率
        """
        adjusted = []
        for token, prob in candidates:
            length_ratio = len(token) / target_length
            adjusted_prob = prob * math.exp(-beta * (length_ratio - 1))
            adjusted.append((token, adjusted_prob))
        
        total = sum(prob for _, prob in adjusted)
        return [(token, prob/total) for token, prob in adjusted]
    
    def _generate_combinations(self, tokens: List[str], target_length: int) -> List[Tuple[str, ...]]:
        """
        生成符合目标长度的组合
        """
        valid_combinations = []
        for r in range(1, len(tokens) + 1):
            for combo in combinations(tokens, r):
                if sum(len(token) for token in combo) == target_length:
                    valid_combinations.append(combo)
        return valid_combinations
    
    def _find_best_combination(self, combinations: List[Tuple[str, ...]], token_probs: List[Tuple[str, float]]) -> Tuple[str, ...]:
        """
        找出最佳组合
        """
        if not combinations:
            return None
            
        best_prob = 0
        best_combo = None
        
        for combo in combinations:
            prob = 1.0
            for token in combo:
                token_prob = next((p for t, p in token_probs if t == token), 0)
                prob *= token_prob
            
            if prob > best_prob:
                best_prob = prob
                best_combo = combo
        
        return best_combo

def save_to_json(output_file: str, assistant_input: str, steps: List[str], result: str, char_count: int, is_correct: bool):
    """
    保存结果到JSON文件
    """
    data = {
        "assistant_input": assistant_input,
        "steps": steps,
        "result": result,
        "char_count": char_count,
        "is_correct": is_correct
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_last_completed_number(output_dir: str, model_name: str) -> int:
    """
    获取上次完成的数字
    通过检查输出文件的数量来确定是否完成了某个数字的处理
    返回最后完成的数字，如果没有完成的数字则返回0
    """
    completed = 0
    for i in range(1, 21):
        # 计算当前数字的输出文件数量
        file_pattern = f"{model_name}_output_{i}_*.json"
        files = [f for f in os.listdir(output_dir) if f.startswith(f"{model_name}_output_{i}_") and f.endswith('.json')]
        
        # 如果文件数量达到1000，说明这个数字处理完成
        if len(files) >= 1000:
            completed = i
            logging.info(f"数字 {i} 已完成，找到 {len(files)} 个输出文件")
        else:
            logging.info(f"数字 {i} 未完成，只找到 {len(files)} 个输出文件")
            break
            
    return completed

def main():
    logging.info("开始运行主程序...")
    
    # 确保输出目录存在
    output_dir = "improve/result/llm_length_improver"
    stats_dir = "improve/result/llm_length_improver/stats"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    logging.info(f"输出目录已创建: {output_dir}")
    
    improver = LLMLengthImprover()
    model_name = "deepseek-r1-250120"  # 添加模型名称
    
    # 获取上次完成的数字
    last_completed = get_last_completed_number(output_dir, model_name)
    logging.info(f"上次完成到数字: {last_completed}")
    
    try:
        with open('data/chinese/简单生成任务.txt', 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
            logging.info(f"成功读取输入文件，共 {len(lines)} 行")
            
            # 从上次完成的数字的下一个开始
            for i in range(last_completed + 1, 21):
                logging.info(f"开始处理数字 {i} 的句子...")
                # 初始化统计数据
                total = 0
                true = 0
                false = 0
                
                for sentence in tqdm(lines, desc=f"Processing sentences for number {i}", unit="sentence"):
                    try:
                        # 从原始句子中提取目标字数
                        numbers = re.findall(r'\d+', sentence)
                        if not numbers:
                            logging.warning(f"无法从句子中提取字数: {sentence}")
                            continue
                        target_length = int(numbers[0])
                        
                        # 替换句子中的数字为当前的i
                        assistant_input = sentence
                        for num in numbers:
                            assistant_input = re.sub(str(num), str(i), assistant_input, 1)
                        logging.info(f"处理句子: {assistant_input}")
                        logging.info(f"目标字数: {i}")  # 使用替换后的数字作为目标长度
                        
                        # 指定个数的例子提示
                        with open("data/prompt/number_case.txt", 'r') as prompt_file:
                            prompt_lines = [line.strip() for line in prompt_file]
                            prompt = prompt_lines[i-1]  # 使用对应数字的提示
                        
                        response = improver.get_llm_response(assistant_input, i)  # 使用i作为目标长度
                        if not response:
                            logging.warning("API返回空响应")
                            continue
                            
                        char_count = len(re.findall(r'[\u4e00-\u9fff]', response))
                        is_correct = char_count == i  # 使用i作为目标长度
                        
                        logging.info(f"生成的句子：{response}")
                        logging.info(f"生成的字数：{char_count}")
                        logging.info(f"字数是否正确：{is_correct}")
                        
                        # 更新统计数据
                        total += 1
                        if is_correct:
                            true += 1
                        else:
                            false += 1
                        
                        # 保存结果
                        output_file = os.path.join(output_dir, f"{model_name}_output_{i}_{total}.json")
                        save_to_json(
                            output_file,
                            assistant_input,
                            [],  # 空列表，因为llm_length_improver没有步骤记录
                            response,
                            char_count,
                            is_correct
                        )
                    except Exception as e:
                        logging.error(f"处理句子时出错: {str(e)}")
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
                
                logging.info(f"\n数字 {i} 的统计结果：")
                logging.info(f"总数：{total}")
                logging.info(f"正确数：{true}")
                logging.info(f"错误数：{false}")
                logging.info(f"正确率：{stats['accuracy']:.2%}")
                logging.info(f"错误率：{stats['error_rate']:.2%}")
                
    except Exception as e:
        logging.error(f"程序运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()

 