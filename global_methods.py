# import openai
import numpy as np
import json
import time
import sys
import os

# import google.generativeai as genai
# from anthropic import Anthropic
import requests


# def get_openai_embedding(texts, model="text-embedding-ada-002"):
#    texts = [text.replace("\n", " ") for text in texts]
#    return np.array([openai.Embedding.create(input = texts, model=model)['data'][i]['embedding'] for i in range(len(texts))])

# def set_anthropic_key():
#     pass

# def set_gemini_key():

#     # Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
#     genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# def set_openai_key():
#     openai.api_key = os.environ['OPENAI_API_KEY']


# def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
#                 model='davinci', use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):

#     run_loop = True
#     counter = 0
#     while run_loop:
#         try:
#             if examples is not None and input is not None:
#                 output = run_chatgpt_with_examples(query, examples, input, num_gen=num_gen, wait_time=wait_time,
#                                                    num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature).strip()
#             else:
#                 output = run_chatgpt(query, num_gen=num_gen, wait_time=wait_time, model=model,
#                                                    num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature)
#             output = output.replace('json', '') # this frequently happens
#             facts = json.loads(output.strip())
#             run_loop = False
#         except json.decoder.JSONDecodeError:
#             counter += 1
#             time.sleep(1)
#             print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
#             print(output)
#             if counter == 10:
#                 print("Exiting after 10 trials")
#                 sys.exit()
#             continue
#     return facts


# def run_claude(query, max_new_tokens, model_name):

#     if model_name == 'claude-sonnet':
#         model_name = "claude-3-sonnet-20240229"
#     elif model_name == 'claude-haiku':
#         model_name = "claude-3-haiku-20240307"

#     client = Anthropic(
#     # This is the default and can be omitted
#     api_key=os.environ.get("ANTHROPIC_API_KEY"),
#     )
#     # print(query)
#     message = client.messages.create(
#         max_tokens=max_new_tokens,
#         messages=[
#             {
#                 "role": "user",
#                 "content": query,
#             }
#         ],
#         model=model_name,
#     )
#     print(message.content)
#     return message.content[0].text


# def run_gemini(model, content: str, max_tokens: int = 0):

#     try:
#         response = model.generate_content(content)
#         return response.text
#     except Exception as e:
#         print(f'{type(e).__name__}: {e}')
#         return None


def run_ollama(query, num_tokens_request=1000, model='qwen2.5:3b',
               temperature=1.0, wait_time=1, host='http://localhost:11434', max_retries=5):
    """
    调用本地 Ollama 模型

    Args:
        query: 查询文本
        num_tokens_request: 请求的最大token数
        model: Ollama 模型名称,默认为 'qwen2.5:3b'
        temperature: 温度参数
        wait_time: 重试等待时间
        host: Ollama API 地址,默认为 localhost:11434
        max_retries: 最大重试次数

    Returns:
        模型生成的文本
    """
    url = f"{host}/api/generate"

    payload = {
        "model": model,
        "prompt": query,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_tokens_request
        }
    }

    completion = None
    current_wait = wait_time
    retry_count = 0

    while completion is None and retry_count < max_retries:
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            completion = result.get('response', '').strip()

            # 检查是否为空响应
            if not completion:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"警告: Ollama 返回空响应,已重试 {max_retries} 次")
                    return ""  # 返回空字符串而不是抛出异常

                print(f"Ollama 返回空响应,重试 {retry_count}/{max_retries}...")
                time.sleep(current_wait)
                current_wait = min(current_wait * 2, 60)  # 最多等待60秒
                completion = None  # 重置以继续循环
                continue

        except requests.exceptions.ConnectionError as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"错误: 无法连接到 Ollama 服务,已重试 {max_retries} 次")
                print(f"请确保 Ollama 已启动并运行在 {host}")
                raise Exception(f"Ollama 服务连接失败: {e}")

            print(f"无法连接到 Ollama 服务 (尝试 {retry_count}/{max_retries})")
            print(f"等待 {current_wait} 秒后重试...")
            time.sleep(current_wait)
            current_wait = min(current_wait * 2, 60)

        except requests.exceptions.Timeout as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"错误: Ollama 请求超时,已重试 {max_retries} 次")
                raise Exception(f"Ollama 请求超时: {e}")

            print(f"Ollama 请求超时 (尝试 {retry_count}/{max_retries})")
            print(f"等待 {current_wait} 秒后重试...")
            time.sleep(current_wait)
            current_wait = min(current_wait * 2, 60)

        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"错误: Ollama API 请求失败,已重试 {max_retries} 次")
                raise Exception(f"Ollama API 请求错误: {e}")

            print(f"Ollama API 请求错误 (尝试 {retry_count}/{max_retries}): {e}")
            print(f"等待 {current_wait} 秒后重试...")
            time.sleep(current_wait)
            current_wait = min(current_wait * 2, 60)

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"错误: 未知错误,已重试 {max_retries} 次")
                raise Exception(f"Ollama 未知错误: {e}")

            print(f"发生未知错误 (尝试 {retry_count}/{max_retries}): {e}")
            print(f"等待 {current_wait} 秒后重试...")
            time.sleep(current_wait)
            current_wait = min(current_wait * 2, 60)

    if completion is None:
        print(f"警告: 达到最大重试次数 ({max_retries}),返回空字符串")
        return ""

    return completion


# def run_chatgpt(query, num_gen=1, num_tokens_request=1000, 
#                 model='chatgpt', use_16k=False, temperature=1.0, wait_time=1):

#     completion = None
#     while completion is None:
#         wait_time = wait_time * 2
#         try:
#             # if model == 'davinci':
#             #     completion = openai.Completion.create(
#             #                     # model = "gpt-3.5-turbo",
#             #                     model = "text-davinci-003",
#             #                     temperature = temperature,
#             #                     max_tokens = num_tokens_request,
#             #                     n=num_gen,
#             #                     prompt=query
#             #                 )
#             if model == 'chatgpt':
#                 messages = [
#                         {"role": "system", "content": query}
#                     ]
#                 completion = openai.ChatCompletion.create(
#                     model="gpt-3.5-turbo",
#                     temperature = temperature,
#                     max_tokens = num_tokens_request,
#                     n=num_gen,
#                     messages = messages
#                 )
#             elif 'gpt-4' in model:
#                 completion = openai.ChatCompletion.create(
#                     model=model,
#                     temperature = temperature,
#                     max_tokens = num_tokens_request,
#                     n=num_gen,
#                     messages = [
#                         {"role": "user", "content": query}
#                     ]
#                 )
#             else:
#                 print("Did not find model %s" % model)
#                 raise ValueError
#         except openai.error.APIError as e:
#             #Handle API error here, e.g. retry or log
#             print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
#             time.sleep(wait_time)
#             pass
#         except openai.error.APIConnectionError as e:
#             #Handle connection error here
#             print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
#             time.sleep(wait_time)
#             pass
#         except openai.error.RateLimitError as e:
#             #Handle rate limit error (we recommend using exponential backoff)
#             print(f"OpenAI API request exceeded rate limit: {e}")
#             pass
#         except openai.error.ServiceUnavailableError as e:
#             #Handle rate limit error (we recommend using exponential backoff)
#             print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
#             time.sleep(wait_time)
#             pass
#         # except Exception as e:
#         #     if e:
#         #         print(e)
#         #         print(f"Timeout error, retrying after waiting for {wait_time} seconds")
#         #         time.sleep(wait_time)
    

#     if model == 'davinci':
#         outputs = [choice.get('text').strip() for choice in completion.get('choices')]
#         if num_gen > 1:
#             return outputs
#         else:
#             # print(outputs[0])
#             return outputs[0]
#     else:
#         # print(completion.choices[0].message.content)
#         return completion.choices[0].message.content
    

# def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000, use_16k=False, wait_time = 1, temperature=1.0):

#     completion = None
    
#     messages = [
#         {"role": "system", "content": query}
#     ]
#     for inp, out in examples:
#         messages.append(
#             {"role": "user", "content": inp}
#         )
#         messages.append(
#             {"role": "system", "content": out}
#         )
#     messages.append(
#         {"role": "user", "content": input}
#     )   
    
#     while completion is None:
#         wait_time = wait_time * 2
#         try:
#             completion = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
#                 temperature = temperature,
#                 max_tokens = num_tokens_request,
#                 n=num_gen,
#                 messages = messages
#             )
#         except openai.error.APIError as e:
#             #Handle API error here, e.g. retry or log
#             print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
#             time.sleep(wait_time)
#             pass
#         except openai.error.APIConnectionError as e:
#             #Handle connection error here
#             print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
#             time.sleep(wait_time)
#             pass
#         except openai.error.RateLimitError as e:
#             #Handle rate limit error (we recommend using exponential backoff)
#             print(f"OpenAI API request exceeded rate limit: {e}")
#             pass
#         except openai.error.ServiceUnavailableError as e:
#             #Handle rate limit error (we recommend using exponential backoff)
#             print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
#             time.sleep(wait_time)
#             pass
    
#     return completion.choices[0].message.content
