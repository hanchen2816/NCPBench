import re
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI

def get_response(messages: List[Dict[str, str]], model_name: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None) -> Any:
    """
    调用大模型API获取响应
    Args:
        messages: 消息列表
        model_name: 模型名称
        api_key: API密钥
        base_url: API基础URL
    Returns:
        模型响应
    """
    # 添加1秒延迟，避免请求过于频繁
    time.sleep(1)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.8,
        top_p=0.8
    )

def LLM(assistant_input: str, is_chinese: bool = True, model_name: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """
    通用LLM调用函数
    Args:
        assistant_input: 输入内容
        is_chinese: 是否使用中文
        model_name: 模型名称
        api_key: API密钥
        base_url: API基础URL
    Returns:
        生成的文本内容
    """
    system_prompt = "你是一个智能助手，能够根据用户的需求生成内容，请严格遵守用户的要求，包括内容数量、质量、格式等规则，如果出现数字，要完全按照数字字数生成，不要多也不要少一个字。输出格式为一句话。" if is_chinese else "You are an intelligent assistant that can generate content according to the needs of users, please strictly follow the user's requirements, including rules such as the quantity, quality, and format of content, and if there are numbers, they should be generated exactly according to the number of words, not more and not less than one word. The output format is one sentence."
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': ("请根据下面要求，生成对应的文段，要求其中除去符号后，汉字的个数必须跟要求的一致" if is_chinese else "Please generate the corresponding paragraph according to the following requirements, and the number of words after the place symbol must be consistent with the requirements.") + str(assistant_input)}
    ]
    try:
        completion = get_response(messages, model_name, api_key, base_url)
        if not hasattr(completion, 'choices') or not completion.choices:
            return "抱歉，模型未返回有效的内容。"
        return completion.choices[0].message.content
    except Exception as e:
        print(f"发生错误: {e}")
        return "抱歉，发生了错误，无法完成请求。" 