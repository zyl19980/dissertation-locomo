#!/usr/bin/env python3
"""
测试扩展评估指标

运行这个脚本来验证所有扩展指标是否正常工作
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试依赖导入"""
    print("="*60)
    print("测试依赖导入")
    print("="*60)

    success = True

    # 测试基础依赖
    try:
        import nltk
        print("✓ nltk 已安装")
    except ImportError:
        print("✗ nltk 未安装: pip install nltk")
        success = False

    try:
        from rouge import Rouge
        print("✓ rouge-score 已安装")
    except ImportError:
        print("✗ rouge-score 未安装: pip install rouge-score")
        success = False

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers 已安装")
    except ImportError:
        print("✗ sentence-transformers 未安装: pip install sentence-transformers")
        success = False

    # 测试 NLTK 数据
    try:
        from nltk.translate.meteor_score import meteor_score
        print("✓ NLTK METEOR 数据可用")
    except LookupError:
        print("✗ NLTK 数据缺失: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")
        success = False

    print()
    return success


def test_metrics():
    """测试所有指标计算"""
    print("="*60)
    print("测试指标计算")
    print("="*60)

    from task_eval.extended_metrics import (
        compute_bleu1,
        compute_rouge2,
        compute_rougel,
        compute_meteor,
        compute_sbert_similarity,
        compute_all_metrics
    )

    # 测试样例
    prediction = "The meeting was very productive"
    ground_truth = "The meeting was productive"

    print(f"预测: {prediction}")
    print(f"参考: {ground_truth}")
    print()

    # 测试 BLEU-1
    try:
        bleu1 = compute_bleu1(prediction, ground_truth)
        print(f"✓ BLEU-1:   {bleu1:.4f}")
    except Exception as e:
        print(f"✗ BLEU-1 计算失败: {e}")
        return False

    # 测试 ROUGE-2
    try:
        rouge2 = compute_rouge2(prediction, ground_truth)
        print(f"✓ ROUGE-2:  {rouge2:.4f}")
    except Exception as e:
        print(f"✗ ROUGE-2 计算失败: {e}")
        return False

    # 测试 ROUGE-L
    try:
        rougel = compute_rougel(prediction, ground_truth)
        print(f"✓ ROUGE-L:  {rougel:.4f}")
    except Exception as e:
        print(f"✗ ROUGE-L 计算失败: {e}")
        return False

    # 测试 METEOR
    try:
        meteor = compute_meteor(prediction, ground_truth)
        print(f"✓ METEOR:   {meteor:.4f}")
    except Exception as e:
        print(f"✗ METEOR 计算失败: {e}")
        return False

    # 测试 SBERT
    try:
        print("\n加载 SBERT 模型 (首次运行会下载模型)...")
        sbert = compute_sbert_similarity(prediction, ground_truth)
        print(f"✓ SBERT:    {sbert:.4f}")
    except Exception as e:
        print(f"✗ SBERT 计算失败: {e}")
        print("提示: 如果网络问题,可以使用 --no-sbert 跳过此指标")

    print()

    # 测试综合计算
    try:
        print("测试综合计算...")
        metrics = compute_all_metrics(prediction, ground_truth)
        print("✓ 所有指标计算成功:")
        for name, value in metrics.items():
            print(f"  {name.upper():10s}: {value:.4f}")
    except Exception as e:
        print(f"✗ 综合计算失败: {e}")
        return False

    print()
    return True


def test_edge_cases():
    """测试边界情况"""
    print("="*60)
    print("测试边界情况")
    print("="*60)

    from task_eval.extended_metrics import compute_all_metrics

    test_cases = [
        ("Empty prediction", "", "reference text"),
        ("Empty reference", "prediction text", ""),
        ("Identical texts", "same text", "same text"),
        ("Completely different", "apple banana", "car house"),
        ("Special characters", "Hello, world!", "Hello world"),
    ]

    for name, pred, ref in test_cases:
        try:
            metrics = compute_all_metrics(pred, ref, sbert_model=None)
            print(f"✓ {name:25s}: BLEU={metrics['bleu1']:.2f}, SBERT={metrics['sbert']:.2f}")
        except Exception as e:
            print(f"✗ {name:25s}: 失败 - {e}")

    print()
    return True


def test_batch_evaluation():
    """测试批量评估"""
    print("="*60)
    print("测试批量评估")
    print("="*60)

    from task_eval.extended_metrics import evaluate_qa_with_extended_metrics

    # 模拟问答数据
    qas = [
        {
            'question': 'What is the capital of France?',
            'answer': 'Paris',
            'test_prediction': 'Paris',
            'category': 1
        },
        {
            'question': 'When did the meeting happen?',
            'answer': 'yesterday',
            'test_prediction': 'yesterday',
            'category': 2
        },
        {
            'question': 'Who attended the event?',
            'answer': 'John, Mary',
            'test_prediction': 'John and Mary',
            'category': 1
        }
    ]

    try:
        print("评估 3 个问答对...")
        results = evaluate_qa_with_extended_metrics(
            qas,
            eval_key='test_prediction',
            use_sbert=False  # 跳过 SBERT 以加快测试
        )

        print(f"✓ 批量评估成功:")
        for metric_name, scores in results.items():
            if len(scores) > 0:
                avg = sum(scores) / len(scores)
                print(f"  {metric_name.upper():10s}: {avg:.4f} (n={len(scores)})")

        print()
        return True

    except Exception as e:
        print(f"✗ 批量评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n")
    print("🧪 扩展评估指标测试")
    print("="*60)
    print()

    all_passed = True

    # 测试依赖
    if not test_imports():
        print("\n⚠️  依赖检查失败,请先安装依赖:")
        print("pip install nltk rouge-score sentence-transformers")
        print("python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")
        return 1

    # 测试指标计算
    if not test_metrics():
        print("\n❌ 指标计算测试失败")
        all_passed = False

    # 测试边界情况
    if not test_edge_cases():
        print("\n❌ 边界情况测试失败")
        all_passed = False

    # 测试批量评估
    if not test_batch_evaluation():
        print("\n❌ 批量评估测试失败")
        all_passed = False

    # 总结
    print("="*60)
    if all_passed:
        print("✅ 所有测试通过!")
        print("="*60)
        print("\n现在可以使用扩展指标:")
        print("python scripts/run_ollama_eval.py --model qwen3:8b --use-extended-metrics")
        return 0
    else:
        print("❌ 部分测试失败")
        print("="*60)
        print("\n请检查上述错误信息并修复")
        return 1


if __name__ == '__main__':
    sys.exit(main())
