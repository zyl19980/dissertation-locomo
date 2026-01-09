import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import os, json
from tqdm import tqdm
import time
from global_methods import run_ollama


MAX_LENGTH={
    'ollama': 8192,
    'qwen2.5:3b': 32768,
    'qwen2.5:7b': 32768,
    'qwen3-8b': 32768,
    'qwen3:8b': 32768,  # Ollama 的标准格式
    'qwen2.5:14b': 32768,
    'qwen2.5:32b': 32768,
    'qwen2.5:72b': 32768,
}
PER_QA_TOKEN_BUDGET = 50

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words.
Write the answers in the form of a json dictionary where each entry contains the question number as "key" and the short answer as "value".
Use single-quote characters for named entities and double-quote characters for enclosing json elements. Answer with exact words from the conversations whenever possible.

"""

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is written at the beginning of the conversation.\n\n"


def process_ouput(text):
    """处理 Ollama 模型的 JSON 输出"""
    single_quote_count = text.count("'")
    double_quote_count = text.count('"')
    if single_quote_count > double_quote_count:
        text = text.replace('"', "")
        text = text.replace("'", '"')

    answers = json.loads(text)
    if type(answers) == dict:
        for k, v in answers.items():
            if v is None:
                answers[k] = ""
                continue
            if isinstance(v, str) and v.startswith('{') and v.endswith('}'):
                try:
                    answers[k] = json.loads(v)['answer']
                except:
                    continue
        return answers
    elif type(answers) == list:
        for k, v in enumerate(answers):
            if v is None:
                answers[k] = ""
            if isinstance(v, str) and v.startswith('{') and v.endswith('}'):
                try:
                    answers[k] = json.loads(v)['answer']
                except:
                    continue
    return answers


def get_cat_5_answer(model_prediction, answer_key):
    """处理类别5的多选题答案"""
    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        if 'a' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    elif len(model_prediction) == 3:
        if '(a)' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    else:
        return model_prediction


def get_input_context(data, num_question_tokens, args):
    """获取输入上下文,尽可能多地包含会话历史"""
    query_conv = ''
    min_session = -1
    stop = False

    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]

    # 简化的 token 估计 (每个字符约0.3个token)
    estimated_tokens_per_char = 0.3
    max_context_length = MAX_LENGTH.get(args.model, 8192)

    for i in range(min(session_nums), max(session_nums) + 1):
        if 'session_%s' % i in data:
            session_text = "\n\n"
            for dialog in data['session_%s' % i][::-1]:
                turn = dialog['speaker'] + ' said, "' + dialog['text'] + '"\n'
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'

                # 估计 token 数量
                estimated_tokens = len(session_text + turn + query_conv) * estimated_tokens_per_char

                if (estimated_tokens + num_question_tokens) < (max_context_length - (PER_QA_TOKEN_BUDGET * args.batch_size)):
                    session_text = turn + session_text
                else:
                    min_session = i
                    stop = True
                    break

            query_conv = 'DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + session_text + query_conv

        if stop:
            break

    return query_conv


def get_ollama_answers(in_data, out_data, prediction_key, args):
    """
    使用 Ollama 本地模型生成问答答案

    Args:
        in_data: 输入数据,包含对话和问题
        out_data: 输出数据,用于保存预测结果
        prediction_key: 预测结果的键名
        args: 参数对象

    Returns:
        out_data: 包含预测结果的输出数据
    """

    assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))

    # 获取对话者名称
    speakers_names = list(set([d['speaker'] for d in in_data['conversation']['session_1']]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = 100  # 估计的起始提示 token 数

    if args.use_rag:
        # RAG 模式需要单独处理
        raise NotImplementedError("RAG mode is not yet implemented for Ollama models")
    else:
        context_database, query_vectors = None, None

    # 遍历所有问题
    for batch_start_idx in tqdm(range(0, len(in_data['qa']), args.batch_size), desc='Generating answers with Ollama'):

        questions = []
        include_idxs = []
        cat_5_idxs = []
        cat_5_answers = []

        for i in range(batch_start_idx, batch_start_idx + args.batch_size):

            if i >= len(in_data['qa']):
                break

            qa = in_data['qa'][i]

            # 检查是否需要生成预测
            if prediction_key not in out_data['qa'][i] or args.overwrite:
                include_idxs.append(i)
            else:
                print(f"Skipping question: {qa['question']}")
                continue

            # 根据问题类别构造提示
            if qa['category'] == 2:
                questions.append(qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.')
            elif qa['category'] == 5:
                # 多选题处理
                question = qa['question'] + " Select the correct answer: (a) {} (b) {}. "
                if random.random() < 0.5:
                    question = question.format('Not mentioned in the conversation', qa['answer'])
                    answer = {'a': 'Not mentioned in the conversation', 'b': qa['answer']}
                else:
                    question = question.format(qa['answer'], 'Not mentioned in the conversation')
                    answer = {'b': 'Not mentioned in the conversation', 'a': qa['answer']}

                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
            else:
                questions.append(qa['question'])


        if questions == []:
            continue

        # 构造查询上下文
        context_ids = None
        if args.use_rag:
            raise NotImplementedError("RAG mode is not yet implemented for Ollama models")
        else:
            question_prompt = QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
            num_question_tokens = len(question_prompt) * 0.3  # 简单估计
            query_conv = get_input_context(in_data['conversation'], num_question_tokens + start_tokens, args)
            query_conv = start_prompt + query_conv

        # 控制调用频率,避免过载
        if args.batch_size > 1:
            time.sleep(0.5)

        # 根据 batch_size 选择单问题或批量处理
        if args.batch_size == 1:
            # 单问题处理
            query = query_conv + '\n\n' + QA_PROMPT.format(questions[0]) if len(cat_5_idxs) == 0 else query_conv + '\n\n' + QA_PROMPT_CAT_5.format(questions[0])

            try:
                answer = run_ollama(
                    query=query,
                    num_tokens_request=100,
                    model=args.model if args.model != 'ollama' else 'qwen3:8b',
                    temperature=0,
                    wait_time=2
                )

                if len(cat_5_idxs) > 0:
                    answer = get_cat_5_answer(answer, cat_5_answers[0])

                out_data['qa'][include_idxs[0]][prediction_key] = answer.strip()
                if args.use_rag:
                    out_data['qa'][include_idxs[0]][prediction_key + '_context'] = context_ids

            except Exception as e:
                print(f"Error processing question: {e}")
                out_data['qa'][include_idxs[0]][prediction_key] = ""

        else:
            # 批量处理
            query = query_conv + '\n' + question_prompt

            trials = 0
            answer = None
            while trials < 3:
                try:
                    trials += 1
                    print(f"Trial {trials}/3")

                    answer = run_ollama(
                        query=query,
                        num_tokens_request=args.batch_size * PER_QA_TOKEN_BUDGET,
                        model=args.model if args.model != 'ollama' else 'qwen3-8b',
                        temperature=0,
                        wait_time=2
                    )

                    answer = answer.replace('\\"', "'").replace('json', '').replace('`', '').strip().replace("\\'", "")
                    answers = process_ouput(answer.strip())
                    break

                except json.decoder.JSONDecodeError as e:
                    print(f'JSON decode error at trial {trials}/3: {e}')
                    if trials == 3:
                        print("Failed after 3 trials")
                        # 使用空答案
                        answers = {str(k): "" for k in range(len(include_idxs))}
                except Exception as e:
                    print(f'Error at trial {trials}/3: {e}')
                    if trials == 3:
                        print("Failed after 3 trials")
                        answers = {str(k): "" for k in range(len(include_idxs))}

            # 处理批量答案
            for k, idx in enumerate(include_idxs):
                try:
                    if k in cat_5_idxs:
                        predicted_answer = get_cat_5_answer(answers[str(k)], cat_5_answers[cat_5_idxs.index(k)])
                        out_data['qa'][idx][prediction_key] = predicted_answer
                    else:
                        try:
                            out_data['qa'][idx][prediction_key] = str(answers[str(k)]).replace('(a)', '').replace('(b)', '').strip()
                        except:
                            out_data['qa'][idx][prediction_key] = ', '.join([str(n) for n in list(answers[str(k)].values())])
                except:
                    try:
                        answers_list = json.loads(answer.strip()) if isinstance(answer, str) else answers
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answers_list[k], cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = answers_list[k].replace('(a)', '').replace('(b)', '').strip()
                    except:
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answer.strip() if answer else "", cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = ""

    return out_data
