import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
from tqdm import tqdm
import argparse
from global_methods import set_openai_key, set_anthropic_key, set_gemini_key
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc
from task_eval.gpt_utils import get_gpt_answers
from task_eval.claude_utils import get_claude_answers
from task_eval.gemini_utils import get_gemini_answers
from task_eval.hf_llm_utils import init_hf_model, get_hf_answers
from task_eval.ollama_utils import get_ollama_answers
from task_eval.extended_metrics import evaluate_qa_with_extended_metrics, print_metric_summary

import numpy as np
import google.generativeai as genai

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--use-extended-metrics', action="store_true",
                        help='计算扩展指标 (BLEU-1, ROUGE-2, ROUGE-L, METEOR, SBERT)')
    parser.add_argument('--no-sbert', action="store_true",
                        help='跳过 SBERT 相似度计算 (加快速度)')
    args = parser.parse_args()
    return args


def main():

    # get arguments
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    if 'gpt' in args.model:
        # set openai API key
        set_openai_key()

    elif 'claude' in args.model:
        # set openai API key
        set_anthropic_key()

    elif 'gemini' in args.model:
        # set openai API key
        set_gemini_key()
        if args.model == "gemini-pro-1.0":
            model_name = "models/gemini-1.0-pro-latest"

        gemini_model = genai.GenerativeModel(model_name)

    elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
        hf_pipeline, hf_model_name = init_hf_model(args)

    elif 'ollama' in args.model or 'qwen' in args.model.lower():
        # Ollama 本地模型,无需设置 API key
        print(f"Using Ollama local model: {args.model}")
        print("Make sure Ollama service is running (default: http://localhost:11434)")

    else:
        raise NotImplementedError


    # load conversations
    samples = json.load(open(args.data_file))
    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    model_key = "%s" % args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)
    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}


    for data in samples:

        out_data = {'sample_id': data['sample_id']}
        if data['sample_id'] in out_samples:
            out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()

        if 'gpt' in args.model:
            # get answers for each sample
            answers = get_gpt_answers(data, out_data, prediction_key, args)
        elif 'claude' in args.model:
            answers = get_claude_answers(data, out_data, prediction_key, args)
        elif 'gemini' in args.model:
            answers = get_gemini_answers(gemini_model, data, out_data, prediction_key, args)
        elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
            answers = get_hf_answers(data, out_data, args, hf_pipeline, hf_model_name)
        elif 'ollama' in args.model or 'qwen' in args.model.lower():
            answers = get_ollama_answers(data, out_data, prediction_key, args)
        else:
            raise NotImplementedError

        # evaluate individual QA samples and save the score
        exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)
        for i in range(0, len(answers['qa'])):
            answers['qa'][i][model_key + '_f1'] = round(exact_matches[i], 3)
            if args.use_rag and len(recall) > 0:
                answers['qa'][i][model_key + '_recall'] = round(recall[i], 3)

        out_samples[data['sample_id']] = answers


    with open(args.out_file, 'w') as f:
        json.dump(list(out_samples.values()), f, indent=2)


    analyze_aggr_acc(args.data_file, args.out_file, args.out_file.replace('.json', '_stats.json'),
                model_key, model_key + '_f1', rag=args.use_rag)

    # 计算扩展指标 (如果启用)
    if args.use_extended_metrics:
        print("\n" + "="*60)
        print("计算扩展评估指标...")
        print("="*60)

        # 收集所有问答对
        all_qas = []
        for sample in out_samples.values():
            all_qas.extend(sample['qa'])

        # 计算扩展指标
        use_sbert = not args.no_sbert
        extended_results = evaluate_qa_with_extended_metrics(
            all_qas,
            eval_key=prediction_key,
            use_sbert=use_sbert
        )

        # 打印摘要
        print_metric_summary(extended_results)

        # 将扩展指标添加到每个问答对
        qa_idx = 0
        for sample in out_samples.values():
            for i in range(len(sample['qa'])):
                if qa_idx < len(extended_results['bleu1']):
                    sample['qa'][i][model_key + '_bleu1'] = round(extended_results['bleu1'][qa_idx], 4)
                    sample['qa'][i][model_key + '_rouge2'] = round(extended_results['rouge2'][qa_idx], 4)
                    sample['qa'][i][model_key + '_rougel'] = round(extended_results['rougel'][qa_idx], 4)
                    sample['qa'][i][model_key + '_meteor'] = round(extended_results['meteor'][qa_idx], 4)
                    if use_sbert:
                        sample['qa'][i][model_key + '_sbert'] = round(extended_results['sbert'][qa_idx], 4)
                qa_idx += 1

        # 保存更新后的结果
        with open(args.out_file, 'w') as f:
            json.dump(list(out_samples.values()), f, indent=2)

        # 保存扩展指标统计
        extended_stats = {
            'model': model_key,
            'total_questions': len(all_qas),
            'metrics': {
                'bleu1': {
                    'mean': float(np.mean(extended_results['bleu1'])),
                    'std': float(np.std(extended_results['bleu1'])),
                    'min': float(np.min(extended_results['bleu1'])),
                    'max': float(np.max(extended_results['bleu1']))
                },
                'rouge2': {
                    'mean': float(np.mean(extended_results['rouge2'])),
                    'std': float(np.std(extended_results['rouge2'])),
                    'min': float(np.min(extended_results['rouge2'])),
                    'max': float(np.max(extended_results['rouge2']))
                },
                'rougel': {
                    'mean': float(np.mean(extended_results['rougel'])),
                    'std': float(np.std(extended_results['rougel'])),
                    'min': float(np.min(extended_results['rougel'])),
                    'max': float(np.max(extended_results['rougel']))
                },
                'meteor': {
                    'mean': float(np.mean(extended_results['meteor'])),
                    'std': float(np.std(extended_results['meteor'])),
                    'min': float(np.min(extended_results['meteor'])),
                    'max': float(np.max(extended_results['meteor']))
                }
            }
        }

        if use_sbert:
            extended_stats['metrics']['sbert'] = {
                'mean': float(np.mean(extended_results['sbert'])),
                'std': float(np.std(extended_results['sbert'])),
                'min': float(np.min(extended_results['sbert'])),
                'max': float(np.max(extended_results['sbert']))
            }

        extended_stats_file = args.out_file.replace('.json', '_extended_stats.json')
        with open(extended_stats_file, 'w') as f:
            json.dump(extended_stats, f, indent=2)

        print(f"✓ 扩展指标已保存到: {extended_stats_file}")

    # encoder=tiktoken.encoding_for_model(args.model))


main()

