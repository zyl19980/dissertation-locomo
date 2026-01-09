#!/usr/bin/env python3
"""
调试版本的评估脚本 - 直接运行以查看详细错误
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import os
import json
import traceback
from tqdm import tqdm
import argparse

# 导入所有需要的模块
try:
    from global_methods import set_openai_key, set_anthropic_key, set_gemini_key
    print("✓ 成功导入 global_methods")
except Exception as e:
    print(f"✗ 导入 global_methods 失败: {e}")
    traceback.print_exc()

try:
    from task_eval.evaluation import eval_question_answering
    print("✓ 成功导入 evaluation")
except Exception as e:
    print(f"✗ 导入 evaluation 失败: {e}")
    traceback.print_exc()

try:
    from task_eval.evaluation_stats import analyze_aggr_acc
    print("✓ 成功导入 evaluation_stats")
except Exception as e:
    print(f"✗ 导入 evaluation_stats 失败: {e}")
    traceback.print_exc()

try:
    from task_eval.gpt_utils import get_gpt_answers
    print("✓ 成功导入 gpt_utils")
except Exception as e:
    print(f"✗ 导入 gpt_utils 失败: {e}")
    traceback.print_exc()

try:
    from task_eval.claude_utils import get_claude_answers
    print("✓ 成功导入 claude_utils")
except Exception as e:
    print(f"✗ 导入 claude_utils 失败: {e}")
    traceback.print_exc()

try:
    from task_eval.gemini_utils import get_gemini_answers
    print("✓ 成功导入 gemini_utils")
except Exception as e:
    print(f"✗ 导入 gemini_utils 失败: {e}")
    traceback.print_exc()

try:
    from task_eval.hf_llm_utils import init_hf_model, get_hf_answers
    print("✓ 成功导入 hf_llm_utils")
except Exception as e:
    print(f"✗ 导入 hf_llm_utils 失败: {e}")
    traceback.print_exc()

try:
    from task_eval.ollama_utils import get_ollama_answers
    print("✓ 成功导入 ollama_utils")
except Exception as e:
    print(f"✗ 导入 ollama_utils 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("所有模块导入成功!")
print("="*60 + "\n")


def main():
    """主评估函数"""

    # 简化的参数解析
    data_file = "data/locomo10.json"
    out_file = "outputs/test_ollama_debug.json"
    model = "qwen3:8b"  # 使用正确的 Ollama 格式
    batch_size = 1

    print(f"配置:")
    print(f"  数据文件: {data_file}")
    print(f"  输出文件: {out_file}")
    print(f"  模型: {model}")
    print(f"  批处理大小: {batch_size}")
    print()

    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"✗ 错误: 数据文件不存在: {data_file}")
        return 1
    print(f"✓ 数据文件存在")

    # 创建输出目录
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print(f"✓ 输出目录已创建")
    print()

    # 加载数据
    print("加载数据...")
    try:
        with open(data_file, 'r') as f:
            samples = json.load(f)
        print(f"✓ 成功加载 {len(samples)} 个样本")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        traceback.print_exc()
        return 1

    print()
    print("="*60)
    print(f"开始评估 {model} 模型")
    print("="*60)
    print()

    # 创建参数对象
    class Args:
        def __init__(self):
            self.data_file = data_file
            self.out_file = out_file
            self.model = model
            self.batch_size = batch_size
            self.use_rag = False
            self.rag_mode = ""
            self.top_k = 5
            self.emb_dir = ""
            self.overwrite = False
            self.retriever = "contriever"
            self.use_4bit = False

    args = Args()

    prediction_key = f"{model}_prediction"
    model_key = model

    # 初始化输出样本
    out_samples = {}
    if os.path.exists(out_file):
        try:
            out_samples = {d['sample_id']: d for d in json.load(open(out_file))}
            print(f"✓ 加载了已有的输出文件,包含 {len(out_samples)} 个样本")
        except:
            print("! 无法加载已有输出文件,将从头开始")

    print()

    # 处理每个样本
    try:
        for sample_idx, data in enumerate(samples):
            print(f"\n处理样本 {sample_idx + 1}/{len(samples)}: {data['sample_id']}")

            out_data = {'sample_id': data['sample_id']}
            if data['sample_id'] in out_samples:
                out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
            else:
                out_data['qa'] = data['qa'].copy()

            print(f"  该样本有 {len(data['qa'])} 个问题")

            # 调用 Ollama 生成答案
            try:
                print(f"  调用 Ollama 模型生成答案...")
                answers = get_ollama_answers(data, out_data, prediction_key, args)
                print(f"  ✓ 成功生成答案")
            except Exception as e:
                print(f"  ✗ 生成答案失败: {e}")
                traceback.print_exc()
                # 继续处理下一个样本
                continue

            # 评估答案
            try:
                exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)
                for i in range(0, len(answers['qa'])):
                    answers['qa'][i][model_key + '_f1'] = round(exact_matches[i], 3)
                print(f"  ✓ 评估完成,平均 F1: {sum(exact_matches)/len(exact_matches):.3f}")
            except Exception as e:
                print(f"  ✗ 评估失败: {e}")
                traceback.print_exc()

            out_samples[data['sample_id']] = answers

            # 定期保存
            if (sample_idx + 1) % 5 == 0:
                print(f"\n  保存中间结果...")
                with open(out_file, 'w') as f:
                    json.dump(list(out_samples.values()), f, indent=2)
                print(f"  ✓ 已保存 {len(out_samples)} 个样本")

    except KeyboardInterrupt:
        print("\n\n用户中断,保存已处理的结果...")
    except Exception as e:
        print(f"\n\n✗ 处理过程中出错: {e}")
        traceback.print_exc()

    # 保存最终结果
    print("\n" + "="*60)
    print("保存最终结果...")
    try:
        with open(out_file, 'w') as f:
            json.dump(list(out_samples.values()), f, indent=2)
        print(f"✓ 结果已保存到: {out_file}")
    except Exception as e:
        print(f"✗ 保存结果失败: {e}")
        traceback.print_exc()
        return 1

    # 生成统计
    print("\n生成统计信息...")
    try:
        stats_file = out_file.replace('.json', '_stats.json')
        analyze_aggr_acc(data_file, out_file, stats_file, model_key, model_key + '_f1', rag=False)
        print(f"✓ 统计信息已保存到: {stats_file}")
    except Exception as e:
        print(f"✗ 生成统计失败: {e}")
        traceback.print_exc()

    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n程序异常终止: {e}")
        traceback.print_exc()
        sys.exit(1)
