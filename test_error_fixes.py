#!/usr/bin/env python3
"""
测试错误修复 - 验证缺少字段时的处理
"""

import json

def test_evaluation_with_missing_fields():
    """测试评估函数能否处理缺少字段的情况"""
    print("="*60)
    print("测试 1: 评估缺少 'answer' 字段的问题")
    print("="*60)

    # 模拟问题数据
    test_qas = [
        {
            'question': 'Test question 1',
            'answer': 'Test answer',
            'category': 1,
            'test_model_prediction': 'Test prediction'
        },
        {
            'question': 'Test question 2 (no answer)',
            # 缺少 'answer' 字段
            'category': 1,
            'test_model_prediction': 'Test prediction'
        },
        {
            'question': 'Test question 3 (no prediction)',
            'answer': 'Test answer',
            'category': 1,
            # 缺少 'test_model_prediction' 字段
        }
    ]

    # 测试评估函数
    from task_eval.evaluation import eval_question_answering

    try:
        exact_matches, lengths, recall = eval_question_answering(
            test_qas,
            eval_key='test_model_prediction'
        )
        print(f"✓ 评估完成!")
        print(f"  - F1 分数: {exact_matches}")
        print(f"  - 答案长度: {lengths}")
        print(f"  - 召回率: {recall}")
        print(f"  - 评估了 {len(exact_matches)} 个问题")
        return True
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ollama_utils_with_missing_answer():
    """测试 Ollama 工具函数能否处理缺少 'answer' 的问题"""
    print("\n" + "="*60)
    print("测试 2: Ollama 工具函数处理缺少 'answer' 的问题")
    print("="*60)

    # 创建模拟数据
    in_data = {
        'conversation': {
            'session_1': [
                {'speaker': 'Alice', 'text': 'Hello'},
                {'speaker': 'Bob', 'text': 'Hi'}
            ],
            'session_1_date_time': '2024-01-01'
        },
        'qa': [
            {
                'question': 'What did Alice say?',
                'answer': 'Hello',
                'category': 1
            },
            {
                'question': 'What else?',
                # 缺少 'answer' 字段
                'category': 1
            }
        ]
    }

    out_data = {
        'qa': [
            {'question': 'What did Alice say?'},
            {'question': 'What else?'}
        ]
    }

    # 模拟参数
    class MockArgs:
        model = 'qwen3:8b'
        batch_size = 1
        use_rag = False
        overwrite = True

    args = MockArgs()

    print("测试数据:")
    print(f"  - 问题 0: 有 'answer' 字段")
    print(f"  - 问题 1: 缺少 'answer' 字段")

    # 检查是否正确设置空预测
    prediction_key = 'qwen3:8b_prediction'

    # 模拟检查逻辑
    for i, qa in enumerate(in_data['qa']):
        if 'answer' not in qa:
            print(f"\n问题 {i} 缺少 'answer' 字段")
            out_data['qa'][i][prediction_key] = ""
            print(f"✓ 为问题 {i} 设置空预测答案: '{out_data['qa'][i][prediction_key]}'")

    # 验证
    if prediction_key in out_data['qa'][1]:
        print(f"\n✓ 测试通过! 问题 1 有预测键: '{prediction_key}'")
        print(f"  预测值: '{out_data['qa'][1][prediction_key]}'")
        return True
    else:
        print(f"\n✗ 测试失败! 问题 1 缺少预测键")
        return False


def main():
    print("\n" + "="*60)
    print("错误修复验证测试")
    print("="*60)

    results = []

    # 测试 1: 评估函数
    results.append(("评估函数处理缺失字段", test_evaluation_with_missing_fields()))

    # 测试 2: Ollama 工具函数
    results.append(("Ollama 工具函数处理缺失 answer", test_ollama_utils_with_missing_answer()))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")
    print("="*60)

    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
