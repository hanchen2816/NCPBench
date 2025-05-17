import argparse
import os
from range_test import test

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='运行范围约束测试脚本')
    
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
    
    # 根据语言选择运行测试
    is_chinese = args.language == 'cn'
    
    # 创建输出目录
    output_base = os.path.join('result/range_test', '中文' if is_chinese else '英文', args.output_dir)
    os.makedirs(output_base, exist_ok=True)
    
    print(f"开始运行{'中文' if is_chinese else '英文'}范围约束测试...")
    print(f"使用模型: {args.model_name}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {output_base}")
    if args.base_url:
        print(f"API基础URL: {args.base_url}")
    
    # 运行测试
    test(
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