import re

def is_numeric_constraint_problem(prompt):
    """
    判断输入是否属于数字约束问题。
    """
    numeric_keywords = ["不少于", "不多于", "至少", "最多", "字数", "个字"]
    # 检查是否包含数字或关键词
    if any(kw in prompt for kw in numeric_keywords) or re.search(r'\d+', prompt):
        return True
    return False

def adjust_output_length(output, target_length):
    """
    调整输出的字数，使其满足目标字数要求。
    """
    current_length = len(output)  # 字符长度计算可能需要更精确
    if current_length == target_length:
        return output, 0  # 不需要调整

    if current_length < target_length:
        # 添加字数
        difference = target_length - current_length
        filler = "，".join(["内容补充"] * (difference // 4))  # 假设每句平均4字
        if len(filler) < difference:  # 如果补充句子不足，再补充
            filler += "。" * (difference - len(filler))
        return output + filler[:difference], difference

    else:
        # 删除字数
        difference = current_length - target_length
        trimmed_output = output[:target_length]  # 简单截断
        return trimmed_output, -difference

def process_input_output(prompt, output, target_length):
    """
    根据输入和输出进行处理：
    1. 判断输入是否为数字约束问题。
    2. 对输出进行字数调整。
    """
    # 判断输入类型
    is_numeric = is_numeric_constraint_problem(prompt)
    problem_type = "数字约束问题" if is_numeric else "非数字约束问题"

    # 调整输出字数
    adjusted_output, adjustment = adjust_output_length(output, target_length)

    result = {
        "输入类型": problem_type,
        "原始输出": output,
        "调整后输出": adjusted_output,
        "字数调整": adjustment,
    }
    return result

# 示例测试
if __name__ == "__main__":
    input_prompt = "请生成10个字的风景介绍"
    output_text = "风和日丽。"
    target_length = 10  # 期望输出字数

    result = process_input_output(input_prompt, output_text, target_length)
    for key, value in result.items():
        print(f"{key}: {value}")
