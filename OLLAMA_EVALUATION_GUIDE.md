# Ollama 模型评估指南

本文档介绍如何使用 Ollama 本地模型在 LoCoMo 数据集上进行问答评估。

## 目录

1. [快速开始](#快速开始)
2. [评估脚本使用](#评估脚本使用)
3. [配置选项](#配置选项)
4. [评估结果](#评估结果)
5. [与其他模型对比](#与其他模型对比)
6. [故障排除](#故障排除)

---

## 快速开始

### 前置条件

1. **安装 Ollama**
   ```bash
   # 访问 https://ollama.com/download
   # Windows: 下载安装包并安装
   # Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **下载模型**
   ```bash
   # 推荐使用 Qwen3-8B (性能好,速度快)
   ollama pull qwen3-8b

   # 或其他模型
   ollama pull qwen2.5:7b
   ollama pull qwen2.5:3b  # 更小更快
   ```

3. **启动 Ollama 服务**
   ```bash
   # Ollama 通常会自动在后台运行
   # 如果需要手动启动:
   ollama serve
   ```

4. **验证服务**
   ```bash
   # 检查服务是否运行
   curl http://localhost:11434/api/tags

   # 或使用 Python 测试脚本
   cd e:\1-ntu\dissertation\论文代码\locomo\locomo
   python test_ollama.py
   ```

---

## 评估脚本使用

### 方法 1: 使用 Python 快速启动脚本 (推荐)

这是最简单的方法,适合 Windows、Linux 和 Mac:

```bash
# 切换到项目目录
cd e:\1-ntu\dissertation\论文代码\locomo\locomo

# 使用默认配置运行 (qwen3-8b 模型)
python scripts/run_ollama_eval.py

# 指定不同的模型
python scripts/run_ollama_eval.py --model qwen2.5:3b

# 指定数据文件
python scripts/run_ollama_eval.py --model qwen2.5:3b --data-file data/locomo10.json

# 覆盖已有结果
python scripts/run_ollama_eval.py --model qwen3-8b --overwrite

# 使用更大的批处理大小 (可能不稳定,不推荐)
python scripts/run_ollama_eval.py --model qwen3-8b --batch-size 5

# 查看所有选项
python scripts/run_ollama_eval.py --help
```

### 方法 2: 使用 Windows 批处理脚本

```cmd
cd e:\1-ntu\dissertation\论文代码\locomo\locomo
scripts\evaluate_ollama.bat
```

**修改配置**: 编辑 `scripts/evaluate_ollama.bat` 文件:
```batch
REM 修改这行来使用不同的模型
set OLLAMA_MODEL=qwen3-8b

REM 修改批处理大小
set BATCH_SIZE=1
```

### 方法 3: 使用 Bash 脚本 (Linux/Mac)

```bash
cd e:\1-ntu\dissertation\论文代码\locomo\locomo
bash scripts/evaluate_ollama.sh
```

**修改配置**: 编辑 `scripts/evaluate_ollama.sh` 文件:
```bash
# 修改这行来使用不同的模型
OLLAMA_MODEL="qwen3-8b"

# 修改批处理大小
BATCH_SIZE=1
```

### 方法 4: 直接调用评估脚本

最灵活的方法,可以自定义所有参数:

```bash
cd e:\1-ntu\dissertation\论文代码\locomo\locomo

python task_eval/evaluate_qa.py \
    --data-file data/locomo10.json \
    --out-file outputs/ollama_qwen3_8b_qa.json \
    --model qwen3-8b \
    --batch-size 1
```

---

## 配置选项

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data-file` | LoCoMo 数据文件路径 | `data/locomo10.json` |
| `--out-file` | 输出结果文件路径 | `outputs/ollama_qwen3_8b_qa.json` |
| `--model` | Ollama 模型名称 | `qwen3-8b`, `qwen2.5:7b` |

### 可选参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--batch-size` | 每批处理的问题数 | 1 | 1 (更稳定) |
| `--overwrite` | 覆盖已有的预测结果 | False | 按需使用 |
| `--use-rag` | 启用 RAG 检索增强 | False | 暂不推荐 |
| `--rag-mode` | RAG 模式类型 | dialog | dialog/summary/observation |
| `--top-k` | RAG 检索数量 | 5 | 3-10 |
| `--emb-dir` | 嵌入向量目录 | outputs | outputs |

### 支持的模型

代码中已配置的模型及其上下文长度:

| 模型名称 | 上下文长度 | 推荐用途 |
|---------|-----------|----------|
| `qwen3-8b` | 32,768 tokens | **推荐**: 平衡性能和速度 |
| `qwen2.5:7b` | 32,768 tokens | 性能好,稍慢 |
| `qwen2.5:3b` | 32,768 tokens | 速度快,适合快速测试 |
| `qwen2.5:14b` | 32,768 tokens | 更好的性能,需要更多资源 |
| `ollama` | 8,192 tokens | 通用占位符,默认使用 qwen3-8b |

---

## 评估结果

### 输出文件

评估完成后,会生成两个文件:

1. **结果文件** (`outputs/ollama_<model>_<dataset>_qa.json`)
   - 包含所有问题和模型的预测答案
   - 格式与其他模型评估结果相同

2. **统计文件** (`outputs/ollama_<model>_<dataset>_qa_stats.json`)
   - 包含评估指标和统计数据
   - F1 分数、准确率等

### 结果示例

```json
{
  "sample_id": "sample_001",
  "qa": [
    {
      "question": "What did John say about the meeting?",
      "answer": "It was productive",
      "qwen3-8b_prediction": "It was productive",
      "qwen3-8b_f1": 1.0,
      "category": 1
    }
  ]
}
```

### 查看统计结果

```bash
# 查看统计文件
cat outputs/ollama_qwen3_8b_locomo10_qa_stats.json

# 或使用 Python
python -c "import json; print(json.dumps(json.load(open('outputs/ollama_qwen3_8b_locomo10_qa_stats.json')), indent=2))"
```

---

## 与其他模型对比

### 评估相同数据集的多个模型

```bash
# 评估 Qwen3-8B
python scripts/run_ollama_eval.py --model qwen3-8b

# 评估 Qwen2.5-7B
python scripts/run_ollama_eval.py --model qwen2.5:7b

# 评估 Qwen2.5-3B
python scripts/run_ollama_eval.py --model qwen2.5:3b
```

### 与 API 模型对比

```bash
# Ollama 本地模型
python scripts/run_ollama_eval.py --model qwen3-8b

# GPT-3.5
bash scripts/evaluate_gpts.sh  # 需要配置 OPENAI_API_KEY

# Gemini
bash scripts/evaluate_gemini.sh  # 需要配置 GOOGLE_API_KEY

# Claude
bash scripts/evaluate_claude.sh  # 需要配置 ANTHROPIC_API_KEY
```

### 模型对比表

| 特性 | Ollama (本地) | GPT-3.5 (API) | Gemini Pro (API) | Claude (API) |
|-----|--------------|---------------|------------------|--------------|
| **成本** | 免费 | 按使用付费 | 按使用付费 | 按使用付费 |
| **隐私** | 完全本地 | 数据上传 | 数据上传 | 数据上传 |
| **速度** | 取决于硬件 | 快 | 快 | 快 |
| **性能** | 中等偏上 | 高 | 高 | 高 |
| **需求** | 本地算力 | API 密钥 | API 密钥 | API 密钥 |

---

## 故障排除

### 问题 1: 无法连接到 Ollama 服务

**症状**:
```
无法连接到 Ollama 服务: Connection refused
```

**解决方案**:
1. 检查 Ollama 是否安装:
   ```bash
   ollama --version
   ```

2. 启动 Ollama 服务:
   ```bash
   ollama serve
   ```

3. 验证服务运行:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### 问题 2: 模型未找到

**症状**:
```
Error: model 'qwen3-8b' not found
```

**解决方案**:
```bash
# 下载模型
ollama pull qwen3-8b

# 查看已安装的模型
ollama list
```

### 问题 3: 内存不足

**症状**:
- 系统变慢
- Ollama 进程崩溃
- 出现 OOM 错误

**解决方案**:
1. 使用更小的模型:
   ```bash
   python scripts/run_ollama_eval.py --model qwen2.5:3b
   ```

2. 关闭其他占用内存的程序

3. 降低批处理大小 (已经是 1,无法再降低)

### 问题 4: 响应超时

**症状**:
```
Ollama 请求超时: timeout
```

**解决方案**:
1. 首次加载模型会较慢,请耐心等待
2. 检查系统资源是否充足
3. 在代码中增加超时时间 (默认 300 秒)

### 问题 5: JSON 解析错误

**症状**:
```
JSON decode error: Expecting value
```

**原因**: 模型输出格式不符合预期

**解决方案**:
1. 使用 `--batch-size 1` (单问题模式更稳定)
2. 尝试不同的模型
3. 检查模型输出格式

### 问题 6: 评估速度慢

**优化建议**:
1. 使用更小的模型 (`qwen2.5:3b`)
2. 确保 Ollama 模型已加载到内存中
3. 升级硬件 (更多 RAM、更快的 CPU/GPU)
4. 使用 `--batch-size` > 1 (但可能不稳定)

---

## 高级用法

### 自定义 Ollama 服务地址

如果 Ollama 运行在非默认端口或远程服务器上:

修改 `global_methods.py` 中的 `run_ollama` 函数调用:

```python
# 在 ollama_utils.py 中
answer = run_ollama(
    query=query,
    num_tokens_request=100,
    model=args.model,
    temperature=0,
    wait_time=2,
    host='http://your-server:11434'  # 添加自定义地址
)
```

### 调整模型参数

在 `ollama_utils.py` 中修改 `run_ollama` 调用:

```python
answer = run_ollama(
    query=query,
    num_tokens_request=100,  # 增加生成长度
    model=args.model,
    temperature=0.7,  # 调整温度 (0-1)
    wait_time=2
)
```

### 批量评估多个模型

```bash
#!/bin/bash
# 评估多个 Ollama 模型

MODELS=("qwen2.5:3b" "qwen2.5:7b" "qwen3-8b")

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    python scripts/run_ollama_eval.py --model $model
    echo "Done with $model"
    echo "---"
done
```

---

## 总结

使用 Ollama 评估的主要优势:
- ✅ 完全免费,无 API 成本
- ✅ 数据隐私,本地处理
- ✅ 离线工作,不依赖网络
- ✅ 与其他模型评估结果格式一致

推荐工作流程:
1. 使用 `qwen2.5:3b` 快速测试代码
2. 使用 `qwen3-8b` 进行正式评估
3. 比较不同模型的结果
4. 根据需要调整参数和配置

如有问题,请参考 [OLLAMA_SETUP.md](OLLAMA_SETUP.md) 或提交 Issue。
