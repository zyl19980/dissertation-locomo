# Ollama 本地模型集成 - 修改总结

## 📋 已添加/修改的文件

### 核心功能文件

1. **`global_methods.py`**
   - ✅ 添加了 `run_ollama()` 函数
   - 支持本地 Ollama API 调用
   - 包含错误处理和重试机制

2. **`task_eval/ollama_utils.py`** (新建)
   - ✅ 完整的 Ollama 评估工具函数
   - 支持单问题和批量处理
   - 处理不同类别的问题

3. **`task_eval/evaluate_qa.py`**
   - ✅ 添加了 Ollama 模型支持
   - 自动检测并调用 `get_ollama_answers()`

4. **`task_eval/gpt_utils.py`**
   - ✅ 导入 `run_ollama`
   - ✅ 在 `MAX_LENGTH` 中添加 Ollama 模型配置
   - ✅ 在 `get_gpt_answers()` 中添加 Ollama 调用逻辑

### 评估脚本

5. **`scripts/run_ollama_eval.py`** (新建)
   - ✅ Python 跨平台评估启动脚本
   - 自动检查 Ollama 服务
   - 支持命令行参数配置

6. **`scripts/evaluate_ollama.sh`** (新建)
   - ✅ Linux/Mac Bash 评估脚本

7. **`scripts/evaluate_ollama.bat`** (新建)
   - ✅ Windows 批处理评估脚本

8. **`scripts/compare_models.py`** (新建)
   - ✅ 模型结果对比工具

### 测试和文档

9. **`test_ollama.py`** (新建)
   - ✅ Ollama 连接测试脚本

10. **`debug_evaluate.py`** (新建)
    - ✅ 调试版评估脚本,显示详细错误信息

11. **`OLLAMA_SETUP.md`** (新建)
    - ✅ Ollama 安装和配置文档

12. **`OLLAMA_EVALUATION_GUIDE.md`** (新建)
    - ✅ 完整的评估使用指南

## 🔧 重要修正

### 模型名称格式
- ❌ 之前使用: `qwen3-8b`
- ✅ 正确格式: `qwen3:8b` (Ollama 标准格式)

已更新所有文件中的默认模型名称。

### 支持的模型

| 模型名称 | 上下文长度 | 状态 |
|---------|-----------|------|
| `qwen3:8b` | 32,768 | ✅ 已配置 |
| `qwen2.5:3b` | 32,768 | ✅ 已配置 |
| `qwen2.5:7b` | 32,768 | ✅ 已配置 |
| `qwen2.5:14b` | 32,768 | ✅ 已配置 |
| `ollama` | 8,192 | ✅ 通用占位符 |

## 🚀 使用方法

### 方法 1: Python 脚本 (推荐)

```bash
# 默认配置 (qwen3:8b)
python scripts/run_ollama_eval.py

# 指定模型
python scripts/run_ollama_eval.py --model qwen2.5:7b

# 调试模式
python debug_evaluate.py
```

### 方法 2: Bash 脚本 (Linux)

```bash
bash scripts/evaluate_ollama.sh
```

### 方法 3: 直接调用

```bash
python task_eval/evaluate_qa.py \
    --data-file data/locomo10.json \
    --out-file outputs/ollama_qwen3_8b_qa.json \
    --model qwen3:8b \
    --batch-size 1
```

## 🐛 调试建议

如果评估失败,请运行调试脚本:

```bash
python debug_evaluate.py 2>&1 | tee debug_output.txt
```

这会显示:
- ✅ 模块导入状态
- ✅ 详细的错误堆栈
- ✅ 每一步的处理进度
- ✅ 完整的异常信息

## ⚠️ 常见问题

### 1. 模型名称错误
**问题**: `qwen3-8b` vs `qwen3:8b`
**解决**: 使用冒号格式 `qwen3:8b`

### 2. Ollama 服务未运行
**检查**:
```bash
curl http://localhost:11434/api/tags
ollama list
```

### 3. 模型未下载
**解决**:
```bash
ollama pull qwen3:8b
```

### 4. 评估失败无输出
**解决**: 运行 `debug_evaluate.py` 查看详细错误

## 📊 输出文件

评估完成后会生成:
- `outputs/ollama_<model>_<dataset>_qa.json` - 预测结果
- `outputs/ollama_<model>_<dataset>_qa_stats.json` - 统计数据

## 🔍 下一步

1. 运行 `python debug_evaluate.py` 查看详细错误
2. 确保模型名称使用正确格式 (`qwen3:8b`)
3. 检查数据文件路径是否正确
4. 验证 Ollama 服务是否运行

## 📝 代码集成说明

Ollama 支持已完全集成到现有评估框架中:
- ✅ 与 GPT、Claude、Gemini 评估脚本一致
- ✅ 支持相同的输出格式
- ✅ 支持相同的评估指标
- ✅ 可直接使用 `compare_models.py` 对比结果
