# 扩展评估指标集成总结

## ✅ 已完成的工作

### 1. 新增文件

| 文件 | 说明 |
|------|------|
| `task_eval/extended_metrics.py` | 扩展指标计算模块 (核心) |
| `test_extended_metrics.py` | 指标测试脚本 |
| `EXTENDED_METRICS_GUIDE.md` | 详细使用指南 |
| `extended_metrics_requirements.txt` | 依赖清单 |
| `scripts/install_extended_metrics_deps.sh` | 依赖安装脚本 |

### 2. 修改文件

| 文件 | 修改内容 |
|------|----------|
| `task_eval/evaluate_qa.py` | 添加扩展指标计算和保存逻辑 |
| `scripts/run_ollama_eval.py` | 添加扩展指标命令行参数 |

### 3. 新增的评估指标

| 指标 | 说明 | 范围 |
|------|------|------|
| **BLEU-1** | 1-gram 词汇重叠 | 0-1 |
| **ROUGE-2** | 2-gram 短语匹配 | 0-1 |
| **ROUGE-L** | 最长公共子序列 | 0-1 |
| **METEOR** | 综合指标(含同义词) | 0-1 |
| **SBERT** | 语义相似度 | 0-1 |

## 🚀 快速开始

### 步骤 1: 安装依赖

```bash
# 方法 1: 使用脚本 (推荐)
bash scripts/install_extended_metrics_deps.sh

# 方法 2: 手动安装
pip install -r extended_metrics_requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 步骤 2: 测试安装

```bash
python test_extended_metrics.py
```

### 步骤 3: 运行评估

```bash
# 完整评估 (包含所有指标)
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics

# 快速评估 (跳过 SBERT)
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics \
    --no-sbert
```

## 📊 输出示例

### 控制台输出

```
============================================================
计算扩展评估指标...
============================================================
加载 SBERT 模型...
✓ SBERT 模型加载成功

============================================================
扩展评估指标摘要
============================================================
BLEU1       : 0.5234 ± 0.2156 (n=150)
ROUGE2      : 0.4123 ± 0.1987 (n=150)
ROUGEL      : 0.5678 ± 0.1823 (n=150)
METEOR      : 0.4567 ± 0.1945 (n=150)
SBERT       : 0.7234 ± 0.1456 (n=150)
============================================================

✓ 扩展指标已保存到: outputs/ollama_qwen3_8b_locomo10_qa_extended_stats.json
```

### 结果文件结构

```
outputs/
├── ollama_qwen3_8b_locomo10_qa.json              # 主结果 (含所有指标)
├── ollama_qwen3_8b_locomo10_qa_stats.json        # 基础统计 (F1)
└── ollama_qwen3_8b_locomo10_qa_extended_stats.json  # 扩展指标统计
```

### JSON 结果示例

```json
{
  "sample_id": "sample_001",
  "qa": [
    {
      "question": "What did John say about the meeting?",
      "answer": "It was productive",
      "qwen3:8b_prediction": "It was very productive",
      "qwen3:8b_f1": 0.857,
      "qwen3:8b_bleu1": 0.6667,
      "qwen3:8b_rouge2": 0.5000,
      "qwen3:8b_rougel": 0.6667,
      "qwen3:8b_meteor": 0.7845,
      "qwen3:8b_sbert": 0.9234
    }
  ]
}
```

## 🎯 使用场景

### 场景 1: 开发阶段 - 快速迭代
```bash
# 只使用基础 F1 指标
python scripts/run_ollama_eval.py --model qwen3:8b
```

### 场景 2: 中期评估 - 词汇指标
```bash
# 添加 BLEU/ROUGE/METEOR,跳过慢速的 SBERT
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics \
    --no-sbert
```

### 场景 3: 最终评估 - 完整指标
```bash
# 包含所有指标,含语义相似度
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics
```

### 场景 4: 批量对比多个模型
```bash
for model in qwen2.5:3b qwen2.5:7b qwen3:8b; do
    python scripts/run_ollama_eval.py \
        --model $model \
        --use-extended-metrics \
        --no-sbert
done
```

## 💡 最佳实践

### 1. 指标选择

| 目标 | 推荐指标 |
|------|---------|
| 词汇准确性 | BLEU-1, F1 |
| 短语匹配 | ROUGE-2 |
| 结构相似性 | ROUGE-L |
| 综合质量 | METEOR |
| 语义相似度 | SBERT |

### 2. 性能优化

- **开发阶段**: 使用 `--no-sbert` 节省时间
- **小数据集** (<100 问题): 包含所有指标
- **大数据集** (>500 问题): 考虑跳过 SBERT 或使用 GPU

### 3. 结果解读

- **BLEU-1 高,SBERT 低**: 用词相似但语义不同
- **BLEU-1 低,SBERT 高**: 表达不同但语义相近 (更好!)
- **所有指标都高**: 优秀的答案
- **所有指标都低**: 需要改进

## 📈 指标对比表

| 指标 | 关注点 | 优点 | 缺点 | 速度 |
|------|--------|------|------|------|
| **F1** | 词汇重叠 | 简单快速 | 忽略词序 | ⚡⚡⚡ |
| **BLEU-1** | 单词匹配 | 标准指标 | 不考虑语义 | ⚡⚡⚡ |
| **ROUGE-2** | 词对匹配 | 捕获短语 | 对长度敏感 | ⚡⚡ |
| **ROUGE-L** | 子序列 | 考虑词序 | 不考虑语义 | ⚡⚡ |
| **METEOR** | 综合 | 含同义词 | 计算复杂 | ⚡ |
| **SBERT** | 语义 | 深层理解 | 计算慢 | 🐌 |

## 🔧 代码集成

### 在自己的代码中使用

```python
from task_eval.extended_metrics import compute_all_metrics

# 单个问答对
prediction = "The meeting was productive"
ground_truth = "The meeting was very productive"

metrics = compute_all_metrics(prediction, ground_truth)

print(f"BLEU-1: {metrics['bleu1']:.4f}")
print(f"ROUGE-2: {metrics['rouge2']:.4f}")
print(f"ROUGE-L: {metrics['rougel']:.4f}")
print(f"METEOR: {metrics['meteor']:.4f}")
print(f"SBERT: {metrics['sbert']:.4f}")
```

### 批量评估

```python
from task_eval.extended_metrics import (
    evaluate_qa_with_extended_metrics,
    print_metric_summary
)

# qas 是问答列表
results = evaluate_qa_with_extended_metrics(
    qas,
    eval_key='model_prediction',
    use_sbert=True
)

# 打印统计
print_metric_summary(results)

# 获取平均分数
import numpy as np
avg_bleu1 = np.mean(results['bleu1'])
avg_sbert = np.mean(results['sbert'])
```

## 📦 依赖说明

### 核心依赖

```
nltk>=3.8.1              # BLEU, METEOR
rouge-score>=0.1.2       # ROUGE-2, ROUGE-L
sentence-transformers>=2.2.2  # SBERT
torch>=2.0.0             # PyTorch (SBERT 后端)
```

### 可选依赖

- GPU 加速: `torch` 的 CUDA 版本
- 更快的 SBERT 模型: 可以替换为其他 Sentence Transformers 模型

## 🐛 常见问题

### Q1: 如何跳过 SBERT?
**A**: 添加 `--no-sbert` 参数

### Q2: SBERT 太慢怎么办?
**A**:
- 使用 `--no-sbert` 跳过
- 使用 GPU (如果有)
- 使用更小的 SBERT 模型

### Q3: NLTK 数据下载失败?
**A**:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Q4: 能否单独使用某个指标?
**A**: 可以,直接导入相应函数:
```python
from task_eval.extended_metrics import compute_bleu1
score = compute_bleu1(pred, ref)
```

## 📚 参考资源

- 详细使用指南: [EXTENDED_METRICS_GUIDE.md](EXTENDED_METRICS_GUIDE.md)
- Ollama 集成文档: [OLLAMA_INTEGRATION_SUMMARY.md](OLLAMA_INTEGRATION_SUMMARY.md)
- 基础评估: [evaluation.py](task_eval/evaluation.py)

## 🎓 学术引用

如果在论文中使用这些指标,请引用:

- **BLEU**: Papineni et al. (2002)
- **ROUGE**: Lin (2004)
- **METEOR**: Banerjee & Lavie (2005)
- **SBERT**: Reimers & Gurevych (2019)

## ✅ 验收检查清单

完成集成后,请确认:

- [ ] 依赖已安装: `pip list | grep -E "nltk|rouge|sentence"`
- [ ] NLTK 数据已下载: `python -c "from nltk.corpus import wordnet; print('OK')"`
- [ ] 测试脚本通过: `python test_extended_metrics.py`
- [ ] 能运行评估: `python scripts/run_ollama_eval.py --model qwen3:8b --use-extended-metrics --no-sbert`
- [ ] 输出文件正确生成: 检查 `outputs/` 目录

## 🎉 总结

现在你的评估系统支持:
- ✅ 基础 F1 分数
- ✅ BLEU-1 词汇匹配
- ✅ ROUGE-2 短语匹配
- ✅ ROUGE-L 结构相似性
- ✅ METEOR 综合指标
- ✅ SBERT 语义相似度

**享受更全面的模型评估!** 🚀
