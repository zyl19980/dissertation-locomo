# 快速修复参考 - KeyError 问题解决方案

## 问题描述

运行评估时遇到两种 KeyError:
1. `KeyError: 'answer'` - 数据中某些问题缺少 answer 字段
2. `KeyError: 'qwen2.5:3b_prediction'` - 评估时找不到预测键

## 根本原因

**问题链**: 缺少 'answer' → 跳过问题 → 没有生成预测 → 评估时找不到预测键 → 崩溃

```
问题 152: 缺少 'answer'
   ↓
ollama_utils.py: 跳过该问题,不生成预测
   ↓
evaluate_qa.py: 调用评估函数
   ↓
evaluation.py: 尝试访问 'qwen2.5:3b_prediction'
   ↓
KeyError: 'qwen2.5:3b_prediction' ❌
```

## 修复方案

### 方案: 双层防护

#### 第一层防护 (ollama_utils.py)
**确保每个问题都有预测键**

```python
# 修改前 (会导致问题):
if 'answer' not in qa:
    include_idxs.pop()  # 跳过,不生成预测 ❌
    continue

# 修改后 (正确):
if 'answer' not in qa:
    out_data['qa'][i][prediction_key] = ""  # 设置空预测 ✓
    continue
```

**位置**: `task_eval/ollama_utils.py:176-188`

#### 第二层防护 (evaluation.py)
**评估前检查键是否存在**

```python
# 新增检查:
if eval_key not in line:
    print(f"警告: 问题 {i} 缺少预测键,跳过评估")
    all_ems.append(0)  # 给 0 分
    all_recall.append(0)
    continue

if 'answer' not in line:
    print(f"警告: 问题 {i} 缺少 answer 字段,跳过评估")
    all_ems.append(0)
    all_recall.append(0)
    continue
```

**位置**: `task_eval/evaluation.py:200-213`

## 修复效果

### 修复前
```
警告: 问题 152 缺少 'answer' 字段,跳过
...
Traceback (most recent call last):
  File "task_eval/evaluate_qa.py", line 112, in main
    exact_matches, lengths, recall = eval_question_answering(...)
  File "task_eval/evaluation.py", line 199, in eval_question_answering
    if type(line[eval_key]) == list:
KeyError: 'qwen2.5:3b_prediction'  ❌ 程序崩溃
```

### 修复后
```
警告: 问题 152 缺少 'answer' 字段,设置空预测答案
...
警告: 问题 152 缺少预测键 'qwen2.5:3b_prediction',跳过评估
199 QA samples evaluated; 199 accuracy values
✓ 评估完成,不会崩溃  ✓
```

## 测试验证

```bash
# 运行测试脚本
python test_error_fixes.py

# 运行完整评估
python scripts/run_ollama_eval.py --model qwen2.5:3b
```

## 相关文件

| 文件 | 修改内容 | 行号 |
|------|----------|------|
| `task_eval/ollama_utils.py` | 设置空预测而不是跳过 | 176-188 |
| `task_eval/evaluation.py` | 添加键存在性检查 | 200-213 |
| `ERROR_FIX_SUMMARY.md` | 详细错误修复文档 | - |
| `test_error_fixes.py` | 测试脚本 | - |

## 关键要点

1. **永远不要完全跳过问题** - 每个问题都应该有预测键,即使是空字符串
2. **防御性编程** - 在访问字典键之前检查其是否存在
3. **优雅降级** - 遇到问题时给 0 分继续,而不是崩溃
4. **详细日志** - 记录所有警告,便于调试

## 数据完整性检查

如果想检查你的数据集中有多少问题缺少 'answer' 字段:

```python
import json

data = json.load(open('data/locomo10.json'))
missing_answer_count = 0

for sample in data:
    for i, qa in enumerate(sample['qa']):
        if 'answer' not in qa:
            print(f"样本 {sample['sample_id']}, 问题 {i} 缺少 answer")
            missing_answer_count += 1

print(f"\n总计: {missing_answer_count} 个问题缺少 'answer' 字段")
```

## 常见问题

**Q: 为什么给缺少 answer 的问题设置空预测?**
A: 因为后续评估代码会遍历所有问题并尝试访问预测键。如果不设置,就会导致 KeyError。

**Q: 评分为 0 是否会影响整体结果?**
A: 会,但这是正确的。缺少 answer 的问题本质上无法评估,给 0 分是合理的。

**Q: 能否直接从数据集中删除这些问题?**
A: 可以,但需要在数据预处理阶段完成。修改代码让其能处理不完整数据更加健壮。

---

**修复完成后,评估应该能够顺利运行,不会因为缺少字段而崩溃!** ✨
