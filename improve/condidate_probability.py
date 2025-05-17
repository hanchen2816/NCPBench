import math
from itertools import combinations
import jieba
import jieba.posseg as pseg

# 定义词性搭配规则
POS_COLLOCATION_RULES = {
    'n': ['v', 'a', 'n'],  # 名词可以搭配动词、形容词、名词
    'v': ['n', 'r', 'd'],  # 动词可以搭配名词、代词、副词
    'a': ['n', 'v', 'd'],  # 形容词可以搭配名词、动词、副词
    'd': ['v', 'a'],       # 副词可以搭配动词、形容词
    'r': ['v', 'n'],       # 代词可以搭配动词、名词
}

def get_pos(word, pos_dict):
    """获取词语的词性"""
    return pos_dict.get(word, 'x')[0]

def check_collocation(word1, word2, pos_dict):
    """检查两个词是否符合搭配规则"""
    pos1 = get_pos(word1, pos_dict)
    pos2 = get_pos(word2, pos_dict)
    return pos2 in POS_COLLOCATION_RULES.get(pos1, [])

def extract_candidates(sentence):
    """
    从原始句子中自动提取候选词并分配概率
    
    Args:
        sentence: 原始句子
    
    Returns:
        候选词及其概率的列表和词性字典
    """
    # 使用jieba进行分词和词性标注
    words = pseg.cut(sentence)
    candidates = []
    pos_dict = {}  # 存储词性信息
    word_weights = {
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
    
    # 收集词语及其权重
    words_with_weights = []
    for word, flag in words:
        if word.strip():  # 忽略空白字符
            weight = word_weights.get(flag[0], 0.5)  # 使用词性的第一个字符作为key
            words_with_weights.append((word, weight))
            pos_dict[word] = flag
    
    # 计算概率
    total_weight = sum(weight for _, weight in words_with_weights)
    candidates = [(word, weight/total_weight) for word, weight in words_with_weights]
    
    # 添加单字候选词（从双字词中提取）
    single_chars = set()
    for word, _ in candidates:
        if len(word) >= 2:
            for char in word:
                single_chars.add(char)
    
    # 为单字设置较低的概率
    base_single_char_prob = 0.1
    for char in single_chars:
        candidates.append((char, base_single_char_prob))
        pos_dict[char] = 'x'  # 为单字设置默认词性
    
    # 重新归一化概率
    total_prob = sum(prob for _, prob in candidates)
    candidates = [(word, prob/total_prob) for word, prob in candidates]
    
    return candidates, pos_dict

def adjust_probability(candidates, target_length, beta=2.0):
    """
    调整候选词的概率分布以生成指定长度的句子
    
    Args:
        candidates: 包含(token, probability)元组的列表
        target_length: 目标句子长度
        beta: 长度惩罚因子
    
    Returns:
        调整后的概率分布列表
    """
    adjusted_probs = []
    for token, prob in candidates:
        token_length = len(token)
        length_ratio = token_length / target_length
        # 使用图中的公式计算调整后的概率
        adjusted_prob = prob * math.exp(-beta * (length_ratio - 1))
        adjusted_probs.append((token, adjusted_prob))
    
    # 归一化概率
    total_prob = sum(prob for _, prob in adjusted_probs)
    normalized_probs = [(token, prob/total_prob) for token, prob in adjusted_probs]
    
    return normalized_probs

def generate_combinations(tokens, target_length):
    """
    生成所有可能的词语组合
    
    Args:
        tokens: 候选词列表
        target_length: 目标长度
    
    Returns:
        符合目标长度的所有可能组合
    """
    valid_combinations = []
    for r in range(1, len(tokens) + 1):
        for combo in combinations(tokens, r):
            total_length = sum(len(token) for token in combo)
            if total_length == target_length:
                valid_combinations.append(combo)
    return valid_combinations

def calculate_combination_probability(combination, token_probs, pos_dict):
    """
    计算组合的总概率，考虑词性搭配
    """
    total_prob = 1.0
    for i, token in enumerate(combination):
        # 基础概率
        prob = next(p for t, p in token_probs if t == token)
        
        # 考虑词性搭配
        if i > 0:
            prev_token = combination[i-1]
            if check_collocation(prev_token, token, pos_dict):
                prob *= 1.5  # 提高符合搭配规则的概率
        
        total_prob *= prob
    
    return total_prob

def process_sentence(original_sentence, target_length):
    """
    处理原始句子，生成指定长度的新句子
    
    Args:
        original_sentence: 原始句子
        target_length: 目标长度
    """
    # 自动提取候选词
    candidates, pos_dict = extract_candidates(original_sentence)
    
    print(f"原始句子：{original_sentence}")
    print(f"目标长度：{target_length}个字")
    
    print("\n提取的候选词：")
    for token, prob in candidates:
        print(f"Token: {token}, Initial Probability: {prob:.3f}")
    
    # 调整概率
    adjusted_probs = adjust_probability(candidates, target_length)
    
    # 生成所有可能的组合
    tokens = [token for token, _ in candidates]
    valid_combinations = generate_combinations(tokens, target_length)
    
    # 计算每个组合的概率并排序
    combination_probs = []
    for combo in valid_combinations:
        prob = calculate_combination_probability(combo, adjusted_probs, pos_dict)
        combination_probs.append((combo, prob))
    
    # 按概率降序排序
    combination_probs.sort(key=lambda x: x[1], reverse=True)
    
    print("\n生成的句子（按概率排序）：")
    for i, (combo, prob) in enumerate(combination_probs[:5], 1):
        sentence = ''.join(combo)
        print(f"{i}. {sentence} (概率: {prob:.4f})")

def main():
    original = "春风轻拂过每一寸土，带来了无尽的生机与希望。"
    process_sentence(original, 4)

if __name__ == "__main__":
    main()
