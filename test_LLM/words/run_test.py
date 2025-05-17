import argparse
import os
from LLM_test_conbine import test

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='运行LLM测试脚本')
    
    # 添加命令行参数
    parser.add_argument('--language', type=str, required=True, choices=['cn', 'en'],
                      help='选择语言：cn-中文，en-英文')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='数据文件目录路径')
    parser.add_argument('--model_name', type=str, required=True,
                      help='大模型名称')
    parser.add_argument('--api_key', type=str, required=True,
                      help='API密钥')
    parser.add_argument('--base_url', type=str,
                      help='API基础URL（可选，用于第三方API服务）')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出目录名称')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['OPENAI_API_KEY'] = args.api_key
    
    # 定义数字类型列表
    number_types = ["素数", "奇数", "偶数", "2的幂", "非2的幂", "完全平方数", "阶乘", "质因数"]
    
    # 根据语言选择运行测试
    is_chinese = args.language == 'cn'
    
    # 创建输出目录
    output_base = os.path.join('result', 'words', '中文' if is_chinese else '英文', args.output_dir)
    os.makedirs(output_base, exist_ok=True)
    
    print(f"开始运行{'中文' if is_chinese else '英文'}测试...")
    print(f"使用模型: {args.model_name}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {output_base}")
    if args.base_url:
        print(f"API基础URL: {args.base_url}")
    
    # 运行测试
    for type in number_types:
        print(f"\n正在测试 {type}...")
        test(
            type=type,
            is_chinese=is_chinese,
            model_name=args.model_name,
            api_key=args.api_key,
            data_dir=args.data_dir,
            output_dir=output_base,
            base_url=args.base_url
        )
    
    print("\n测试完成！")

if __name__ == '__main__':
    main() 