# 扩展评估指标使用指南

本文档说明如何使用新增的扩展评估指标:BLEU-1、ROUGE-2、ROUGE-L、METEOR 和 SBERT 相似度。

## 📦 依赖安装

首先需要安装额外的依赖包:

```bash
# 基础 NLP 指标
pip install nltk rouge-score

# SBERT 语义相似度
pip install sentence-transformers

# 下载 NLTK 数据 (METEOR 需要)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 完整依赖列表

```bash
pip install nltk rouge-score sentence-transformers torch
```

或者使用 requirements 文件:

```txt
# extended_metrics_requirements.txt
nltk>=3.8
rouge-score>=0.1.2
sentence-transformers>=2.2.0
torch>=2.0.0
```

安装:
```bash
pip install -r extended_metrics_requirements.txt
```

## 🚀 使用方法

### 方法 1: 使用 Python 启动脚本 (推荐)

```bash
# 基础评估 + 扩展指标
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics

# 不使用 SBERT (更快)
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics \
    --no-sbert
```

### 方法 2: 直接调用评估脚本

```bash
python task_eval/evaluate_qa.py \
    --data-file data/locomo10.json \
    --out-file outputs/ollama_qwen3_8b_qa.json \
    --model qwen3:8b \
    --batch-size 1 \
    --use-extended-metrics
```

### 方法 3: 仅计算 BLEU/ROUGE/METEOR (跳过 SBERT)

```bash
python task_eval/evaluate_qa.py \
    --data-file data/locomo10.json \
    --out-file outputs/ollama_qwen3_8b_qa.json \
    --model qwen3:8b \
    --batch-size 1 \
    --use-extended-metrics \
    --no-sbert
```

## 📊 评估指标说明

### 1. BLEU-1 (Bilingual Evaluation Understudy)
- **范围**: 0-1 (越高越好)
- **说明**: 测量预测文本和参考文本之间的 1-gram (单词) 重叠
- **用途**: 评估词汇级别的匹配度

### 2. ROUGE-2 (Recall-Oriented Understudy for Gisting Evaluation)
- **范围**: 0-1 (越高越好)
- **说明**: 测量 2-gram (词对) 的 F1 分数
- **用途**: 评估短语级别的匹配度,比 BLEU-1 更严格

### 3. ROUGE-L (Longest Common Subsequence)
- **范围**: 0-1 (越高越好)
- **说明**: 基于最长公共子序列的 F1 分数
- **用途**: 评估句子级别的结构相似性

### 4. METEOR (Metric for Evaluation of Translation with Explicit ORdering)
- **范围**: 0-1 (越高越好)
- **说明**: 考虑同义词、词干化和词序的综合指标
- **用途**: 更全面的文本相似度评估

### 5. SBERT 相似度 (Sentence-BERT Semantic Similarity)
- **范围**: 0-1 (越高越好)
- **说明**: 使用预训练的句子嵌入模型计算语义相似度
- **用途**: 评估深层语义相似性,即使用词不同也能识别相同含义
- **注意**: 计算较慢,但最能反映语义质量

## 📁 输出文件

启用扩展指标后,会生成以下文件:

### 1. 主结果文件 (`*_qa.json`)
包含每个问题的所有指标:
```json
{
  "sample_id": "sample_001",
  "qa": [
    {
      "question": "What did John say?",
      "answer": "It was great",
      "qwen3:8b_prediction": "It was great",
      "qwen3:8b_f1": 1.0,
      "qwen3:8b_bleu1": 1.0,
      "qwen3:8b_rouge2": 0.6667,
      "qwen3:8b_rougel": 1.0,
      "qwen3:8b_meteor": 0.9950,
      "qwen3:8b_sbert": 0.9876
    }
  ]
}
```

### 2. 基础统计文件 (`*_stats.json`)
包含 F1 分数和准确率统计

### 3. 扩展指标统计文件 (`*_extended_stats.json`)
包含所有扩展指标的统计信息:
```json
{
  "model": "qwen3:8b",
  "total_questions": 150,
  "metrics": {
    "bleu1": {
      "mean": 0.5234,
      "std": 0.2156,
      "min": 0.0,
      "max": 1.0
    },
    "rouge2": {
      "mean": 0.4123,
      "std": 0.1987,
      "min": 0.0,
      "max": 1.0
    },
    "rougel": {
      "mean": 0.5678,
      "std": 0.1823,
      "min": 0.0,
      "max": 1.0
    },
    "meteor": {
      "mean": 0.4567,
      "std": 0.1945,
      "min": 0.0,
      "max": 1.0
    },
    "sbert": {
      "mean": 0.7234,
      "std": 0.1456,
      "min": 0.2341,
      "max": 0.9987
    }
  }
}
```

## 💡 使用建议

### 快速测试 (跳过 SBERT)
如果只是快速测试,建议跳过 SBERT:
```bash
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics \
    --no-sbert
```

### 完整评估 (包含 SBERT)
对于最终评估,建议包含所有指标:
```bash
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics
```

### 批量评估多个模型
```bash
#!/bin/bash
MODELS=("qwen2.5:3b" "qwen2.5:7b" "qwen3:8b")

for model in "${MODELS[@]}"; do
    echo "评估 $model..."
    python scripts/run_ollama_eval.py \
        --model $model \
        --use-extended-metrics
done
```

## ⏱️ 性能说明

### 计算时间 (10 个样本,150 个问题)

| 指标 | 计算时间 | 说明 |
|-----|---------|------|
| F1 (基础) | ~1 秒 | 快速 |
| BLEU-1 | ~2 秒 | 快速 |
| ROUGE-2/L | ~3 秒 | 中等 |
| METEOR | ~5 秒 | 中等 |
| SBERT | ~30-60 秒 | 慢 (取决于 GPU) |

**建议**:
- 开发阶段: 使用 `--no-sbert`
- 最终评估: 包含所有指标

## 🔧 高级用法

### 在 Python 代码中直接使用

```python
from task_eval.extended_metrics import compute_all_metrics

prediction = "The meeting was very productive"
ground_truth = "The meeting was productive"

metrics = compute_all_metrics(prediction, ground_truth)

print(f"BLEU-1: {metrics['bleu1']:.4f}")
print(f"ROUGE-2: {metrics['rouge2']:.4f}")
print(f"ROUGE-L: {metrics['rougel']:.4f}")
print(f"METEOR: {metrics['meteor']:.4f}")
print(f"SBERT: {metrics['sbert']:.4f}")
```

### 批量计算

```python
from task_eval.extended_metrics import evaluate_qa_with_extended_metrics, print_metric_summary

# qas 是问答列表
results = evaluate_qa_with_extended_metrics(
    qas,
    eval_key='qwen3:8b_prediction',
    use_sbert=True
)

# 打印摘要
print_metric_summary(results)
```

## 📈 指标对比

不同指标的特点对比:

| 指标 | 词汇匹配 | 短语匹配 | 语义理解 | 同义词 | 速度 |
|-----|---------|---------|---------|--------|-----|
| BLEU-1 | ✅ | ❌ | ❌ | ❌ | 快 |
| ROUGE-2 | ✅ | ✅ | ❌ | ❌ | 快 |
| ROUGE-L | ✅ | ✅ | ⚠️ | ❌ | 快 |
| METEOR | ✅ | ✅ | ⚠️ | ✅ | 中 |
| SBERT | ⚠️ | ⚠️ | ✅ | ✅ | 慢 |

**推荐组合**:
- 快速评估: F1 + BLEU-1 + ROUGE-L
- 全面评估: F1 + BLEU-1 + ROUGE-2 + ROUGE-L + METEOR + SBERT

## 🐛 故障排除

### 问题 1: NLTK 数据缺失
```
LookupError: Resource 'wordnet' not found
```

**解决**:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 问题 2: SBERT 模型下载失败
```
Error downloading model
```

**解决**:
```python
from sentence_transformers import SentenceTransformer

# 手动下载模型
model = SentenceTransformer('all-MiniLM-L6-v2')
```

或使用国内镜像:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题 3: 内存不足 (SBERT)
**解决**: 使用 `--no-sbert` 跳过 SBERT 计算

### 问题 4: ROUGE 计算错误
```
ValueError: Hypothesis is empty
```

这是正常的,代码会自动返回 0.0 分数。

## 📚 参考文献

- **BLEU**: Papineni et al. (2002) - "BLEU: a Method for Automatic Evaluation of Machine Translation"
- **ROUGE**: Lin (2004) - "ROUGE: A Package for Automatic Evaluation of Summaries"
- **METEOR**: Banerjee & Lavie (2005) - "METEOR: An Automatic Metric for MT Evaluation"
- **SBERT**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

## 📞 获取帮助

如果遇到问题,请运行调试脚本:
```bash
python debug_evaluate.py
```

或查看详细日志:
```bash
python task_eval/evaluate_qa.py --help
```
