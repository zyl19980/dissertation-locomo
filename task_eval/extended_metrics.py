"""
扩展的评估指标模块
包含 BLEU-1, ROUGE-2, ROUGE-L, METEOR 和 SBERT 相似度
"""

import numpy as np
import string
import regex
from collections import Counter
from typing import List, Dict
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def normalize_answer(s):
    """标准化答案文本"""
    s = s.replace(',', "")

    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_bleu1(prediction: str, ground_truth: str) -> float:
    """
    计算 BLEU-1 分数

    Args:
        prediction: 预测文本
        ground_truth: 参考文本

    Returns:
        BLEU-1 分数 (0-1)
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        # 标准化并分词
        pred_tokens = normalize_answer(prediction).split()
        ref_tokens = normalize_answer(ground_truth).split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        # 使用平滑函数避免零分
        smoothing = SmoothingFunction().method1

        # BLEU-1 只考虑 1-gram
        score = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothing
        )

        return score

    except Exception as e:
        print(f"BLEU-1 计算错误: {e}")
        return 0.0


def compute_rouge2(prediction: str, ground_truth: str) -> float:
    """
    计算 ROUGE-2 分数 (bigram overlap)

    Args:
        prediction: 预测文本
        ground_truth: 参考文本

    Returns:
        ROUGE-2 F1 分数 (0-1)
    """
    try:
        from rouge import Rouge

        # 标准化
        pred_normalized = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
        ref_normalized = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])

        if not pred_normalized or not ref_normalized:
            return 0.0

        rouge = Rouge()
        scores = rouge.get_scores(pred_normalized, ref_normalized, avg=True)

        return scores["rouge-2"]["f"]

    except Exception as e:
        # print(f"ROUGE-2 计算错误: {e}")
        return 0.0


def compute_rougel(prediction: str, ground_truth: str) -> float:
    """
    计算 ROUGE-L 分数 (longest common subsequence)

    Args:
        prediction: 预测文本
        ground_truth: 参考文本

    Returns:
        ROUGE-L F1 分数 (0-1)
    """
    try:
        from rouge import Rouge

        # 标准化
        pred_normalized = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
        ref_normalized = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])

        if not pred_normalized or not ref_normalized:
            return 0.0

        rouge = Rouge()
        scores = rouge.get_scores(pred_normalized, ref_normalized, avg=True)

        return scores["rouge-l"]["f"]

    except Exception as e:
        # print(f"ROUGE-L 计算错误: {e}")
        return 0.0


def compute_meteor(prediction: str, ground_truth: str) -> float:
    """
    计算 METEOR 分数

    Args:
        prediction: 预测文本
        ground_truth: 参考文本

    Returns:
        METEOR 分数 (0-1)
    """
    try:
        from nltk.translate.meteor_score import meteor_score

        # 标准化并分词
        pred_tokens = normalize_answer(prediction).split()
        ref_tokens = normalize_answer(ground_truth).split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        # METEOR 需要参考文本作为列表
        score = meteor_score([ref_tokens], pred_tokens)

        return score

    except Exception as e:
        print(f"METEOR 计算错误: {e}")
        return 0.0


def compute_sbert_similarity(prediction: str, ground_truth: str, model=None) -> float:
    """
    计算 SBERT (Sentence-BERT) 语义相似度

    Args:
        prediction: 预测文本
        ground_truth: 参考文本
        model: SBERT 模型 (可选,如果为 None 则自动加载)

    Returns:
        余弦相似度 (0-1)
    """
    try:
        from sentence_transformers import SentenceTransformer, util

        # 加载模型 (如果未提供)
        if model is None:
            # 使用轻量级的 SBERT 模型
            model = SentenceTransformer('all-MiniLM-L6-v2')

        # 标准化文本 (保留原始语义,不做太多处理)
        pred_text = prediction.strip()
        ref_text = ground_truth.strip()

        if not pred_text or not ref_text:
            return 0.0

        # 编码
        pred_embedding = model.encode(pred_text, convert_to_tensor=True)
        ref_embedding = model.encode(ref_text, convert_to_tensor=True)

        # 计算余弦相似度
        similarity = util.cos_sim(pred_embedding, ref_embedding).item()

        # 归一化到 [0, 1]
        similarity = max(0.0, min(1.0, similarity))

        return similarity

    except Exception as e:
        print(f"SBERT 相似度计算错误: {e}")
        return 0.0


def compute_all_metrics(prediction: str, ground_truth: str, sbert_model=None) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        prediction: 预测文本
        ground_truth: 参考文本
        sbert_model: SBERT 模型 (可选)

    Returns:
        包含所有指标的字典
    """
    metrics = {
        'bleu1': compute_bleu1(prediction, ground_truth),
        'rouge2': compute_rouge2(prediction, ground_truth),
        'rougel': compute_rougel(prediction, ground_truth),
        'meteor': compute_meteor(prediction, ground_truth),
        'sbert': compute_sbert_similarity(prediction, ground_truth, sbert_model)
    }

    return metrics


def compute_multi_answer_metrics(prediction: str, ground_truths: List[str], sbert_model=None) -> Dict[str, float]:
    """
    对于多个参考答案,计算最佳匹配的指标

    Args:
        prediction: 预测文本
        ground_truths: 参考答案列表
        sbert_model: SBERT 模型 (可选)

    Returns:
        包含所有指标的字典 (取最大值)
    """
    if not ground_truths:
        return {
            'bleu1': 0.0,
            'rouge2': 0.0,
            'rougel': 0.0,
            'meteor': 0.0,
            'sbert': 0.0
        }

    # 对每个参考答案计算指标
    all_metrics = []
    for gt in ground_truths:
        metrics = compute_all_metrics(prediction, gt, sbert_model)
        all_metrics.append(metrics)

    # 取每个指标的最大值
    best_metrics = {
        'bleu1': max(m['bleu1'] for m in all_metrics),
        'rouge2': max(m['rouge2'] for m in all_metrics),
        'rougel': max(m['rougel'] for m in all_metrics),
        'meteor': max(m['meteor'] for m in all_metrics),
        'sbert': max(m['sbert'] for m in all_metrics)
    }

    return best_metrics


def evaluate_qa_with_extended_metrics(qas: List[Dict], eval_key: str = 'prediction',
                                       use_sbert: bool = True) -> Dict[str, List[float]]:
    """
    使用扩展指标评估问答任务

    Args:
        qas: 问答数据列表
        eval_key: 预测结果的键名
        use_sbert: 是否使用 SBERT (计算较慢)

    Returns:
        包含各指标分数列表的字典
    """
    # 加载 SBERT 模型 (如果需要)
    sbert_model = None
    if use_sbert:
        try:
            from sentence_transformers import SentenceTransformer
            print("加载 SBERT 模型...")
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ SBERT 模型加载成功")
        except Exception as e:
            print(f"✗ SBERT 模型加载失败: {e}")
            print("将跳过 SBERT 相似度计算")

    # 初始化结果字典
    results = {
        'bleu1': [],
        'rouge2': [],
        'rougel': [],
        'meteor': [],
        'sbert': []
    }

    # 遍历每个问答对
    for i, qa in enumerate(qas):
        # 获取预测和参考答案
        if eval_key not in qa:
            print(f"警告: 问题 {i} 没有预测结果,跳过")
            continue

        prediction = str(qa[eval_key])

        # 处理答案格式
        if type(qa['answer']) == list:
            ground_truths = qa['answer']
        else:
            ground_truth = str(qa['answer'])

            # 对于类别3的问题,只取第一个答案
            if qa.get('category') == 3:
                ground_truth = ground_truth.split(';')[0].strip()

            # 对于多跳问题(类别1),分割子答案
            if qa.get('category') == 1:
                ground_truths = [gt.strip() for gt in ground_truth.split(',')]
            else:
                ground_truths = [ground_truth]

        # 计算指标
        metrics = compute_multi_answer_metrics(prediction, ground_truths, sbert_model)

        # 保存结果
        for metric_name, score in metrics.items():
            results[metric_name].append(score)

    return results


def print_metric_summary(results: Dict[str, List[float]]):
    """打印指标摘要"""
    print("\n" + "="*60)
    print("扩展评估指标摘要")
    print("="*60)

    for metric_name, scores in results.items():
        if len(scores) > 0:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{metric_name.upper():12s}: {avg_score:.4f} ± {std_score:.4f} (n={len(scores)})")
        else:
            print(f"{metric_name.upper():12s}: N/A")

    print("="*60 + "\n")
