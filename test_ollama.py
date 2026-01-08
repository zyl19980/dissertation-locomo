"""
测试 Ollama 本地模型连接的简单脚本
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from global_methods import run_ollama

def test_ollama_connection():
    """测试 Ollama 连接"""

    print("=" * 60)
    print("测试 Ollama 本地模型连接")
    print("=" * 60)

    # 简单的测试问题
    test_query = "Hello! Please introduce yourself briefly in one sentence."

    print("\n发送测试问题:")
    print(f"  {test_query}")
    print("\n等待 Ollama 响应...\n")

    try:
        # 调用 Ollama (使用 qwen3-8b 模型)
        response = run_ollama(
            query=test_query,
            num_tokens_request=100,
            model='qwen3-8b',  # 你可以改成其他已下载的模型
            temperature=0.7,
            wait_time=2
        )

        print("✓ Ollama 响应成功!")
        print("-" * 60)
        print("模型回复:")
        print(response)
        print("-" * 60)
        print("\n✓ Ollama 连接测试通过!")

    except Exception as e:
        print(f"\n✗ Ollama 连接失败!")
        print(f"错误信息: {e}")
        print("\n请检查:")
        print("1. Ollama 是否已安装")
        print("2. Ollama 服务是否已启动 (运行: ollama serve)")
        print("3. qwen3-8b 模型是否已下载 (运行: ollama pull qwen3-8b)")
        return False

    return True


if __name__ == "__main__":
    test_ollama_connection()
