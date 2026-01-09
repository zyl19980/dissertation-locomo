# LoCoMo Ollama 集成 + 扩展评估指标

本项目为 LoCoMo (Long Context Modeling) 基准测试添加了 **Ollama 本地模型支持** 和 **扩展评估指标**。

## 🎯 新功能

### 1. Ollama 本地模型支持
- ✅ 完全本地运行,无需 API 密钥
- ✅ 支持 Qwen 系列模型 (qwen3:8b, qwen2.5:7b 等)
- ✅ 与现有评估框架无缝集成
- ✅ 支持所有现有的评估模式

### 2. 扩展评估指标
- ✅ **BLEU-1**: 词汇级别匹配
- ✅ **ROUGE-2**: 短语级别匹配
- ✅ **ROUGE-L**: 结构相似性
- ✅ **METEOR**: 综合指标(含同义词)
- ✅ **SBERT**: 语义相似度

## 📦 快速开始

### 前置条件

```bash
# 1. 安装 Ollama
# 访问 https://ollama.com/download

# 2. 下载模型
ollama pull qwen3:8b

# 3. 安装扩展指标依赖 (可选)
pip install -r extended_metrics_requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 运行评估

```bash
# 基础评估 (仅 F1 分数)
python scripts/run_ollama_eval.py --model qwen3:8b

# 完整评估 (包含所有扩展指标)
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics

# 快速评估 (跳过慢速的 SBERT)
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --use-extended-metrics \
    --no-sbert
```

### 测试安装

```bash
# 测试 Ollama 连接
python test_ollama.py

# 测试扩展指标
python test_extended_metrics.py

# 调试评估
python debug_evaluate.py
```

## 📂 项目结构

```
locomo/
├── global_methods.py              # 添加了 run_ollama()
├── task_eval/
│   ├── evaluate_qa.py            # 修改: 支持 Ollama 和扩展指标
│   ├── ollama_utils.py           # 新增: Ollama 工具函数
│   └── extended_metrics.py       # 新增: 扩展评估指标
├── scripts/
│   ├── run_ollama_eval.py        # 新增: Python 评估脚本
│   ├── evaluate_ollama.sh        # 新增: Bash 评估脚本
│   ├── evaluate_ollama.bat       # 新增: Windows 批处理脚本
│   └── compare_models.py         # 新增: 模型对比工具
├── test_ollama.py                # 新增: Ollama 连接测试
├── test_extended_metrics.py      # 新增: 指标测试
├── debug_evaluate.py             # 新增: 调试脚本
└── 文档/
    ├── OLLAMA_SETUP.md           # Ollama 安装指南
    ├── OLLAMA_EVALUATION_GUIDE.md # Ollama 评估指南
    ├── EXTENDED_METRICS_GUIDE.md # 扩展指标使用指南
    ├── OLLAMA_INTEGRATION_SUMMARY.md # Ollama 集成总结
    └── EXTENDED_METRICS_SUMMARY.md   # 扩展指标总结
```

## 🚀 使用示例

### 示例 1: 评估单个模型

```bash
python scripts/run_ollama_eval.py \
    --model qwen3:8b \
    --data-file data/locomo10.json \
    --use-extended-metrics
```

### 示例 2: 批量评估多个模型

```bash
#!/bin/bash
for model in qwen2.5:3b qwen2.5:7b qwen3:8b; do
    echo "评估 $model..."
    python scripts/run_ollama_eval.py \
        --model $model \
        --use-extended-metrics \
        --no-sbert
done
```

### 示例 3: 对比不同模型

```bash
# 评估所有模型
python scripts/run_ollama_eval.py --model qwen2.5:3b
python scripts/run_ollama_eval.py --model qwen3:8b

# 对比结果
python scripts/compare_models.py \
    outputs/ollama_qwen2_5_3b_locomo10_qa.json \
    outputs/ollama_qwen3_8b_locomo10_qa.json
```

## 📊 评估指标说明

| 指标 | 范围 | 说明 | 速度 |
|------|------|------|------|
| **F1** (基础) | 0-1 | 词汇级别的精确率和召回率 | 快 |
| **BLEU-1** | 0-1 | 1-gram 重叠率 | 快 |
| **ROUGE-2** | 0-1 | 2-gram 重叠率 | 快 |
| **ROUGE-L** | 0-1 | 最长公共子序列 | 快 |
| **METEOR** | 0-1 | 综合指标,含同义词匹配 | 中 |
| **SBERT** | 0-1 | 基于 BERT 的语义相似度 | 慢 |

## 📁 输出文件

评估完成后生成以下文件:

```
outputs/
├── ollama_qwen3_8b_locomo10_qa.json          # 主结果文件
├── ollama_qwen3_8b_locomo10_qa_stats.json    # 基础统计
└── ollama_qwen3_8b_locomo10_qa_extended_stats.json  # 扩展指标统计 (如启用)
```

### 结果文件示例

```json
{
  "sample_id": "sample_001",
  "qa": [
    {
      "question": "What did John say?",
      "answer": "It was productive",
      "qwen3:8b_prediction": "It was productive",
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

## 🔧 高级配置

### 自定义 Ollama 服务器

如果 Ollama 运行在非默认端口:

修改 `global_methods.py` 中的 `run_ollama` 函数:

```python
answer = run_ollama(
    query=query,
    model='qwen3:8b',
    host='http://your-server:11434'  # 自定义地址
)
```

### 使用不同的 SBERT 模型

修改 `task_eval/extended_metrics.py`:

```python
# 默认: 'all-MiniLM-L6-v2' (小而快)
model = SentenceTransformer('all-mpnet-base-v2')  # 更大更准确
```

## 📖 完整文档

- **[OLLAMA_SETUP.md](OLLAMA_SETUP.md)** - Ollama 安装和配置
- **[OLLAMA_EVALUATION_GUIDE.md](OLLAMA_EVALUATION_GUIDE.md)** - 详细评估指南
- **[EXTENDED_METRICS_GUIDE.md](EXTENDED_METRICS_GUIDE.md)** - 扩展指标使用指南
- **[OLLAMA_INTEGRATION_SUMMARY.md](OLLAMA_INTEGRATION_SUMMARY.md)** - Ollama 集成总结
- **[EXTENDED_METRICS_SUMMARY.md](EXTENDED_METRICS_SUMMARY.md)** - 扩展指标总结

## ⚡ 性能提示

### 加速评估

1. **跳过 SBERT**: 添加 `--no-sbert` 可节省 80% 时间
2. **使用 GPU**: 安装 CUDA 版本的 PyTorch
3. **减少数据**: 在开发阶段使用较小的数据集

### 时间估算 (150 个问题)

| 配置 | 时间 |
|------|------|
| 基础 F1 | ~2 分钟 |
| + BLEU/ROUGE/METEOR | ~3 分钟 |
| + SBERT (CPU) | ~8 分钟 |
| + SBERT (GPU) | ~4 分钟 |

## 🐛 故障排除

### Ollama 连接失败

```bash
# 检查服务
curl http://localhost:11434/api/tags

# 启动服务
ollama serve

# 查看日志
python debug_evaluate.py
```

### 指标计算错误

```bash
# 测试依赖
python test_extended_metrics.py

# 重新安装
pip install -r extended_metrics_requirements.txt
```

### 内存不足

```bash
# 使用更小的模型
python scripts/run_ollama_eval.py --model qwen2.5:3b

# 跳过 SBERT
python scripts/run_ollama_eval.py --model qwen3:8b --no-sbert
```

## 🌟 特性对比

| 特性 | OpenAI API | Ollama 本地 |
|------|-----------|------------|
| 成本 | 按使用付费 | 免费 |
| 速度 | 快 | 取决于硬件 |
| 隐私 | 数据上传 | 完全本地 |
| 离线使用 | ❌ | ✅ |
| 设置难度 | 简单 | 中等 |
| 模型选择 | 固定 | 灵活 |

## 📞 获取帮助

遇到问题?

1. 查看相应的文档文件
2. 运行测试脚本: `python test_ollama.py` 或 `python test_extended_metrics.py`
3. 运行调试脚本: `python debug_evaluate.py`
4. 检查 GitHub Issues

## 🎉 致谢

- **LoCoMo 原始项目**: 基准测试框架
- **Ollama**: 本地 LLM 运行平台
- **Sentence Transformers**: SBERT 语义相似度模型
- **NLTK & ROUGE**: 文本评估指标

## 📄 许可证

与原 LoCoMo 项目相同的许可证

---

**开始使用**: `python scripts/run_ollama_eval.py --model qwen3:8b --use-extended-metrics` 🚀
