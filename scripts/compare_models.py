#!/usr/bin/env python3
"""
模型结果对比工具

比较不同模型在 LoCoMo QA 任务上的表现
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse


def load_results(file_path: str) -> Dict:
    """加载评估结果文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误: JSON 解析失败 {file_path}")
        return None


def extract_model_name(file_path: str) -> str:
    """从文件名中提取模型名称"""
    filename = Path(file_path).stem
    # 移除常见的后缀
    for suffix in ['_qa', '_stats', '_locomo10', '_locomo']:
        filename = filename.replace(suffix, '')
    return filename


def compare_models(result_files: List[str], output_file: str = None):
    """比较多个模型的结果"""

    results = {}
    for file_path in result_files:
        data = load_results(file_path)
        if data is None:
            continue
        model_name = extract_model_name(file_path)
        results[model_name] = data

    if not results:
        print("错误: 没有成功加载任何结果文件")
        return

    print("=" * 80)
    print("模型结果对比")
    print("=" * 80)
    print()

    # 获取所有模型名称
    model_names = list(results.keys())
    print(f"对比模型: {', '.join(model_names)}")
    print()

    # 假设所有结果文件包含相同的样本
    first_model_data = list(results.values())[0]

    comparison_data = []

    # 遍历每个样本
    for sample_idx, sample in enumerate(first_model_data):
        sample_id = sample.get('sample_id', f'sample_{sample_idx}')

        print(f"样本 {sample_idx + 1}: {sample_id}")
        print("-" * 80)

        # 遍历该样本的每个问题
        for qa_idx, qa in enumerate(sample['qa']):
            question = qa['question']
            gold_answer = qa.get('answer', 'N/A')
            category = qa.get('category', 'N/A')

            print(f"\n  问题 {qa_idx + 1}: {question}")
            print(f"  类别: {category}")
            print(f"  标准答案: {gold_answer}")
            print()

            # 收集每个模型的预测
            for model_name in model_names:
                model_data = results[model_name]

                # 找到对应的样本和问题
                try:
                    model_sample = next(s for s in model_data if s.get('sample_id') == sample_id)
                    model_qa = model_sample['qa'][qa_idx]

                    # 查找预测键
                    prediction_key = None
                    f1_key = None
                    for key in model_qa.keys():
                        if 'prediction' in key and not key.endswith('_context'):
                            prediction_key = key
                        if '_f1' in key:
                            f1_key = key

                    if prediction_key:
                        prediction = model_qa.get(prediction_key, 'N/A')
                        f1_score = model_qa.get(f1_key, 'N/A') if f1_key else 'N/A'
                        print(f"  {model_name:30s}: {prediction:40s} (F1: {f1_score})")
                    else:
                        print(f"  {model_name:30s}: [未找到预测]")

                except (StopIteration, IndexError, KeyError) as e:
                    print(f"  {model_name:30s}: [数据不匹配]")

            print()

        print("=" * 80)
        print()

        # 为简洁起见,只对比前几个样本
        if sample_idx >= 2:  # 只显示前3个样本
            print(f"... (省略剩余 {len(first_model_data) - 3} 个样本)")
            break

    # 统计总体表现
    print()
    print("=" * 80)
    print("总体统计")
    print("=" * 80)

    for model_name in model_names:
        model_data = results[model_name]
        total_questions = 0
        total_f1 = 0
        f1_count = 0

        for sample in model_data:
            for qa in sample['qa']:
                total_questions += 1
                # 查找 F1 分数键
                for key, value in qa.items():
                    if '_f1' in key and isinstance(value, (int, float)):
                        total_f1 += value
                        f1_count += 1
                        break

        avg_f1 = (total_f1 / f1_count) if f1_count > 0 else 0
        print(f"{model_name:30s}: {total_questions} 个问题, 平均 F1 = {avg_f1:.3f}")

    print("=" * 80)

    # 保存对比结果
    if output_file:
        comparison_output = {
            'models': model_names,
            'summary': {}
        }

        for model_name in model_names:
            model_data = results[model_name]
            total_questions = 0
            total_f1 = 0
            f1_count = 0

            for sample in model_data:
                for qa in sample['qa']:
                    total_questions += 1
                    for key, value in qa.items():
                        if '_f1' in key and isinstance(value, (int, float)):
                            total_f1 += value
                            f1_count += 1
                            break

            avg_f1 = (total_f1 / f1_count) if f1_count > 0 else 0
            comparison_output['summary'][model_name] = {
                'total_questions': total_questions,
                'average_f1': round(avg_f1, 3)
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_output, f, indent=2, ensure_ascii=False)

        print()
        print(f"对比结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='比较不同模型的评估结果')
    parser.add_argument('result_files', nargs='+', help='评估结果文件路径')
    parser.add_argument('--output', '-o', help='保存对比结果到文件')

    args = parser.parse_args()

    if len(args.result_files) < 2:
        print("错误: 至少需要提供两个结果文件进行对比")
        return 1

    compare_models(args.result_files, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
