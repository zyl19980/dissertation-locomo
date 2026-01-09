#!/bin/bash
# 安装扩展评估指标所需的依赖

echo "========================================"
echo "安装扩展评估指标依赖"
echo "========================================"
echo ""

# 安装 Python 包
echo "1. 安装 Python 包..."
pip install nltk rouge-score sentence-transformers torch

if [ $? -eq 0 ]; then
    echo "✓ Python 包安装成功"
else
    echo "✗ Python 包安装失败"
    exit 1
fi

echo ""

# 下载 NLTK 数据
echo "2. 下载 NLTK 数据..."
python3 << EOF
import nltk
import sys

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✓ NLTK 数据下载成功")
    sys.exit(0)
except Exception as e:
    print(f"✗ NLTK 数据下载失败: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# 测试安装
echo "3. 测试安装..."
python3 test_extended_metrics.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 所有依赖安装成功!"
    echo "========================================"
    echo ""
    echo "现在可以使用扩展指标:"
    echo "python scripts/run_ollama_eval.py --model qwen3:8b --use-extended-metrics"
else
    echo ""
    echo "========================================"
    echo "✗ 测试失败,请检查错误信息"
    echo "========================================"
    exit 1
fi
