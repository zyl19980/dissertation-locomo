#!/usr/bin/env python3
"""
Ollama 模型评估快速启动脚本

这是一个便捷的 Python 脚本,用于快速运行 Ollama 模型评估。
可以直接运行,也可以修改配置后运行。
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_ollama_service():
    """检查 Ollama 服务是否运行"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✓ Ollama 服务正在运行")
            print(f"✓ 已安装的模型: {[m['name'] for m in models]}")
            return True
        else:
            print(f"✗ Ollama 服务响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 无法连接到 Ollama 服务: {e}")
        print("请确保 Ollama 已安装并运行")
        print("启动命令: ollama serve")
        return False


def main():
    parser = argparse.ArgumentParser(description='Ollama 模型评估脚本')
    parser.add_argument('--model', type=str, default='qwen3-8b',
                        help='Ollama 模型名称 (默认: qwen3-8b)')
    parser.add_argument('--data-file', type=str, default='data/locomo10.json',
                        help='数据文件路径 (默认: data/locomo10.json)')
    parser.add_argument('--out-dir', type=str, default='outputs',
                        help='输出目录 (默认: outputs)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批处理大小 (默认: 1, 推荐保持为1以获得更好的稳定性)')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已有的预测结果')
    parser.add_argument('--skip-check', action='store_true',
                        help='跳过 Ollama 服务检查')
    parser.add_argument('--use-rag', action='store_true',
                        help='使用 RAG 模式 (注意: RAG 模式目前尚未完全支持)')
    parser.add_argument('--rag-mode', type=str, default='dialog',
                        choices=['dialog', 'summary', 'observation'],
                        help='RAG 模式类型')
    parser.add_argument('--top-k', type=int, default=5,
                        help='RAG 模式下检索的 top-k 数量')
    parser.add_argument('--emb-dir', type=str, default='outputs',
                        help='嵌入向量存储目录')

    args = parser.parse_args()

    print("=" * 60)
    print("Ollama 模型评估")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据文件: {args.data_file}")
    print(f"输出目录: {args.out_dir}")
    print(f"批处理大小: {args.batch_size}")
    print("=" * 60)
    print()

    # 检查 Ollama 服务
    if not args.skip_check:
        print("正在检查 Ollama 服务...")
        if not check_ollama_service():
            print("\n错误: Ollama 服务未运行!")
            print("请先启动 Ollama 服务:")
            print("  方法1: 直接运行 'ollama serve'")
            print("  方法2: Ollama 应该会在后台自动运行")
            print("\n如果确定服务已运行,可以使用 --skip-check 参数跳过检查")
            return 1
        print()

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 构建输出文件名
    safe_model_name = args.model.replace(':', '_').replace('.', '_')
    base_filename = os.path.basename(args.data_file).replace('.json', '')
    output_file = os.path.join(args.out_dir, f"ollama_{safe_model_name}_{base_filename}_qa.json")

    # 构建评估命令
    cmd = [
        sys.executable,
        'task_eval/evaluate_qa.py',
        '--data-file', args.data_file,
        '--out-file', output_file,
        '--model', args.model,
        '--batch-size', str(args.batch_size),
    ]

    if args.overwrite:
        cmd.append('--overwrite')

    if args.use_rag:
        cmd.extend([
            '--use-rag',
            '--rag-mode', args.rag_mode,
            '--top-k', str(args.top_k),
            '--emb-dir', args.emb_dir,
        ])

    print("运行评估命令:")
    print(" ".join(cmd))
    print()
    print("=" * 60)
    print("开始评估...")
    print("=" * 60)
    print()

    # 运行评估
    try:
        result = subprocess.run(cmd, cwd=project_root)

        if result.returncode == 0:
            print()
            print("=" * 60)
            print("✓ 评估完成!")
            print("=" * 60)
            print(f"结果文件: {output_file}")
            print(f"统计文件: {output_file.replace('.json', '_stats.json')}")
            print("=" * 60)
            return 0
        else:
            print()
            print("=" * 60)
            print("✗ 评估失败!")
            print("=" * 60)
            return result.returncode

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("评估被用户中断")
        print("=" * 60)
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ 运行出错: {e}")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
