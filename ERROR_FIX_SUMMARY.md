# 错误修复总结 - Ollama 评估健壮性改进

## 🐛 已修复的问题

### 1. KeyError: 'answer' 和 KeyError: 'qwen2.5:3b_prediction'
**问题**:
- 某些问题数据缺少 'answer' 字段
- 当跳过缺少 'answer' 字段的问题后，这些问题没有生成 `prediction_key`
- 后续评估代码尝试访问不存在的键导致 KeyError 崩溃

**修复**:
1. **在 `ollama_utils.py` 中** (lines 176-188):
   - 检查问题是否缺少 'answer' 字段
   - **关键改进**: 对于缺少 'answer' 的问题，设置空预测答案（`""`）而不是完全跳过
   - 这确保每个问题都有 prediction_key，避免后续评估时的 KeyError

```python
# 检查是否需要生成预测
if prediction_key not in out_data['qa'][i] or args.overwrite:
    # 检查是否有必需的字段
    if 'answer' not in qa:
        print(f"警告: 问题 {i} 缺少 'answer' 字段,设置空预测答案")
        # 设置空预测答案,而不是跳过
        out_data['qa'][i][prediction_key] = ""
        continue

    include_idxs.append(i)
```

2. **在 `evaluation.py` 中** (lines 200-213):
   - 在访问 `eval_key` 之前检查其是否存在
   - 如果缺少预测键或 answer 字段，给该问题评分为 0 而不是崩溃
   - 这提供了双重保护

```python
# 检查是否有预测键，如果没有则跳过该问题
if eval_key not in line:
    print(f"警告: 问题 {i} 缺少预测键 '{eval_key}',跳过评估")
    # 添加 0 分以保持索引对应
    all_ems.append(0)
    all_recall.append(0)
    continue

# 检查是否有 answer 字段
if 'answer' not in line:
    print(f"警告: 问题 {i} 缺少 'answer' 字段,跳过评估")
    all_ems.append(0)
    all_recall.append(0)
    continue
```

### 2. Empty response from Ollama
**问题**: Ollama 有时返回空响应,导致程序无限重试

**修复**:
- 在 `global_methods.py` 的 `run_ollama()` 中添加最大重试次数限制
- 默认最大重试 5 次
- 达到重试上限后返回空字符串而不是抛出异常
- 添加指数退避策略,最多等待 60 秒

**代码位置**: `global_methods.py:93-195`

```python
def run_ollama(query, num_tokens_request=1000, model='qwen2.5:3b',
               temperature=1.0, wait_time=1, host='http://localhost:11434', max_retries=5):
    # ...
    retry_count = 0
    while completion is None and retry_count < max_retries:
        try:
            # ...
            if not completion:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"警告: Ollama 返回空响应,已重试 {max_retries} 次")
                    return ""  # 返回空字符串而不是崩溃
                # ...
```

### 3. 未处理的异常导致程序崩溃
**问题**: 各种异常(JSON 解析错误、网络错误等)没有被妥善处理

**修复**:
- 在单问题处理和批量处理中都添加了 try-except 块
- 捕获所有异常,记录错误信息,但继续执行
- 失败的问题使用空字符串作为答案

**代码位置**: `task_eval/ollama_utils.py:236-263, 265-343`

## ✅ 改进的功能

### 1. 智能重试机制

| 特性 | 改进前 | 改进后 |
|------|--------|--------|
| 最大重试 | 无限 | 5 次(可配置) |
| 退避策略 | 线性增长 | 指数退避(最多60秒) |
| 空响应处理 | 抛出异常 | 返回空字符串 |
| 错误信息 | 简单 | 详细(包含重试次数) |

### 2. 错误恢复策略

```
问题发生 → 记录错误 → 使用空答案 → 继续下一个问题
```

这确保了即使部分问题失败,整个评估也能完成。

### 3. 详细的日志输出

现在会输出:
- ✅ 跳过的问题及原因
- ⚠️ 缺少的字段及默认值
- ❌ 失败的请求及重试次数
- 📊 每个步骤的状态

## 🔧 使用建议

### 基本运行
```bash
python scripts/run_ollama_eval.py --model qwen3:8b
```

### 调试模式
如果遇到问题,使用调试脚本查看详细信息:
```bash
python debug_evaluate.py 2>&1 | tee debug_output.txt
```

### 检查日志
查看生成的日志文件,识别问题:
```bash
# 查找错误
grep "错误" debug_output.txt

# 查找警告
grep "警告" debug_output.txt

# 查找跳过的问题
grep "跳过" debug_output.txt
```

## 📊 错误分类和处理

### 1. 数据问题(不会崩溃)
- **缺少 'answer' 字段**: 跳过该问题
- **缺少 'category' 字段**: 使用默认值 1
- **其他缺失字段**: 记录警告,尝试继续

### 2. 网络问题(会重试)
- **连接失败**: 重试 5 次,每次等待时间翻倍
- **超时**: 重试 5 次
- **其他网络错误**: 重试 5 次

### 3. 模型响应问题(会重试)
- **空响应**: 重试 3-5 次,最终返回空字符串
- **JSON 解析失败**: 重试 3 次,最终使用空答案
- **其他格式错误**: 尝试多种解析方法

### 4. 系统错误(会崩溃)
- **Ollama 服务无法启动**: 5 次重试后抛出异常
- **严重的系统错误**: 抛出异常

## 🧪 测试改进

### 测试数据完整性
```python
# 检查数据文件是否有问题
import json

data = json.load(open('data/locomo10.json'))
for sample in data:
    for i, qa in enumerate(sample['qa']):
        if 'answer' not in qa:
            print(f"样本 {sample['sample_id']}, 问题 {i} 缺少 answer")
        if 'category' not in qa:
            print(f"样本 {sample['sample_id']}, 问题 {i} 缺少 category")
```

### 测试 Ollama 连接
```bash
# 测试基本连接
python test_ollama.py

# 测试调用
python -c "from global_methods import run_ollama; print(run_ollama('Hello', model='qwen3:8b'))"
```

## 📈 性能影响

### 重试开销
- **无错误**: 无额外开销
- **偶尔错误**: ~2-10 秒额外时间(重试)
- **频繁错误**: 可能增加 20-50% 的总时间

### 推荐设置
```python
# 快速模式(较少重试)
run_ollama(query, max_retries=3)

# 稳定模式(更多重试) - 默认
run_ollama(query, max_retries=5)

# 最大容错模式
run_ollama(query, max_retries=10)
```

## 🎯 最佳实践

### 1. 运行前检查
```bash
# 检查 Ollama 服务
curl http://localhost:11434/api/tags

# 检查模型
ollama list | grep qwen

# 检查数据文件
python -c "import json; data=json.load(open('data/locomo10.json')); print(f'{len(data)} samples')"
```

### 2. 监控评估
```bash
# 实时查看进度和错误
python scripts/run_ollama_eval.py --model qwen3:8b 2>&1 | tee evaluation.log

# 在另一个终端查看进度
tail -f evaluation.log | grep -E "处理样本|错误|警告"
```

### 3. 处理中断
如果评估中断:
```bash
# 不使用 --overwrite,会自动跳过已完成的问题
python scripts/run_ollama_eval.py --model qwen3:8b

# 强制重新评估
python scripts/run_ollama_eval.py --model qwen3:8b --overwrite
```

## 🔍 故障排查流程

### Step 1: 检查基础环境
```bash
# Ollama 服务
ollama list

# Python 依赖
pip list | grep -E "requests|numpy"
```

### Step 2: 运行调试脚本
```bash
python debug_evaluate.py
```

### Step 3: 查看日志
```bash
# 查找具体错误
cat debug_output.txt | grep -A 5 "Traceback"
```

### Step 4: 针对性修复
- **KeyError**: 检查数据文件格式
- **Empty response**: 检查模型是否正确加载
- **Connection error**: 检查 Ollama 服务状态
- **Timeout**: 增加超时时间或使用更小的模型

## 📝 变更日志

### v1.1.0 (2024-01-09)
- ✅ 添加 `max_retries` 参数到 `run_ollama()`
- ✅ 添加字段存在性检查
- ✅ 改进错误处理,避免崩溃
- ✅ 添加详细的日志输出
- ✅ 添加指数退避策略
- ✅ 改进批量处理的错误恢复

### v1.0.0 (初始版本)
- 基本的 Ollama 集成
- 基础的错误处理

## 🎓 经验总结

### 导致崩溃的常见原因
1. **数据格式不一致**: 某些样本缺少必需字段
2. **网络不稳定**: Ollama 服务间歇性失败
3. **模型行为异常**: 返回空响应或格式错误的输出
4. **资源不足**: 内存或 CPU 限制导致超时

### 解决方案
1. **数据验证**: 在评估前验证数据完整性
2. **容错机制**: 捕获所有异常,使用空答案继续
3. **重试策略**: 智能重试,但有上限
4. **详细日志**: 记录所有问题,便于调试

## 🚀 下一步改进计划

1. 添加数据验证工具
2. 支持断点续传(保存中间结果)
3. 添加评估进度条
4. 支持并行评估(多进程)
5. 添加评估质量检查工具

---

**现在你的评估系统更加健壮,可以处理各种异常情况而不会崩溃!** ✨
