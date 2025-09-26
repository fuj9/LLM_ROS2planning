
from openai import OpenAI
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
import os

# GPU 設定
GPU = 0
if torch.cuda.is_available():
    torch.cuda.set_device(GPU)

# OpenAI クライアント設定
OPENAI_KEY = "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" # 例: export OPENAI_API_KEY="sk-xxxx"
client = None
if OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)

source = 'openai'  # select from ['openai', 'huggingface']
planning_lm_id = 'gpt-4o'  # see comments above for all options
translation_lm_id = 'stsb-xlm-r-multilingual'  # see comments above for all options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.75 # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples

client = OpenAI(api_key=OPENAI_KEY)

if source == 'openai':
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128,
        "n": 5,  # 複数サンプル生成
        # "logprobs": 1,  # ← Chat APIでは未サポートなので削除 or コメントアウト
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop": ["\n"]
    }
elif source == 'huggingface':
    sampling_params = \
            {
              "temperature": 0.7,
              "top_p": 0.9,
              "max_new_tokens": 128
            }

"""### Planning LM Initialization
Initialize **Planning LM** from either **OpenAI API** or **Huggingface Transformers**. Abstract away the underlying source by creating a generator function with a common interface.
"""

def lm_engine(source, planning_lm_id, device):
    if source == 'huggingface':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(planning_lm_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(planning_lm_id, pad_token_id=tokenizer.pad_token_id).to(device)

    def _generate(prompt, sampling_params):
        if source == 'openai':
            response = client.chat.completions.create(
              model=planning_lm_id,  # 例: "gpt-4"
              messages=[{"role": "user", "content": prompt}],
              temperature=sampling_params.get("temperature", 0.7),
              top_p=sampling_params.get("top_p", 0.9),
              max_tokens=sampling_params.get("max_tokens", 128),
              n=sampling_params.get("n", 1),
              presence_penalty=sampling_params.get("presence_penalty", 0.0),
              frequency_penalty=sampling_params.get("frequency_penalty", 0.0),
              stop=sampling_params.get("stop", None),
            )

            generated_samples = [choice.message.content for choice in response.choices]
            mean_log_probs = [0.0 for _ in generated_samples]
            '''
            response = openai.Completion.create(engine=planning_lm_id, prompt=prompt, **sampling_params)
            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
            # calculate mean log prob across tokens
            mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(sampling_params['n'])]
            '''
        elif source == 'huggingface':

            #AddCode
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            prompt_len = input_ids.shape[-1]
            gen_params = {k: v for k, v in sampling_params.items() if k != "max_new_tokens"}
            output_dict = model.generate(input_ids, attention_mask=attention_mask,max_new_tokens=sampling_params.get("max_new_tokens", 128), pad_token_id=tokenizer.pad_token_id,**gen_params)

            # discard the prompt (only take the generated text)
            generated_samples = tokenizer.batch_decode(output_dict.sequences[:, input_ids.shape[-1]:])
            # calculate per-token logprob
            vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)  # [n, length, vocab_size]
            token_log_probs = torch.gather(vocab_log_probs, 2, output_dict.sequences[:, prompt_len:, None]).squeeze(-1).tolist()  # [n, length]
            # truncate each sample if it contains '\n' (the current step is finished)
            # e.g. 'open fridge\n<|endoftext|>' -> 'open fridge'
            for i, sample in enumerate(generated_samples):
                stop_idx = sample.index('\n') if '\n' in sample else None
                generated_samples[i] = sample[:stop_idx]
                token_log_probs[i] = token_log_probs[i][:stop_idx]
            # calculate mean log prob across tokens
            mean_log_probs = [np.mean(token_log_probs[i]) for i in range(sampling_params['num_return_sequences'])]
        generated_samples = [sample.strip().lower() for sample in generated_samples]
        return generated_samples, mean_log_probs

    return _generate

generator = lm_engine(source, planning_lm_id, device="cpu")

"""### Translation LM Initialization
Initialize **Translation LM** and create embeddings for all available actions (for action translation) and task names of all available examples (for finding relevant example)
"""

# initialize Translation LM
translation_lm = SentenceTransformer(translation_lm_id).to(device)

# create action embeddings using Translated LM
with open('available_robotics_actions.json', 'r', encoding="utf-8") as f:
    action_list = json.load(f)
action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory

# create example task embeddings using Translated LM
with open('available_robotics_examples.json', 'r', encoding="utf-8") as f:
    available_examples = json.load(f)
example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
example_task_embedding = translation_lm.encode(example_task_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory

# helper function for finding similar sentence in a corpus given a query
def find_most_similar(query_str, corpus_embedding):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    return most_similar_idx, matching_score

# define query task
# あるもの：ミャクミャク、3Dプリンターの上にある遊び道具、抹茶の箱の中にあるクッキーとお皿
task = "遊び道具を動かして、別の場所に移動して運ぶ"
# find most relevant example
example_idx, _ = find_most_similar(task, example_task_embedding)
example = available_examples[example_idx]
# construct initial prompt
curr_prompt = f'{example}\n\nTask: {task}'
# print example and query task
print('-'*10 + ' GIVEN EXAMPLE ' + '-'*10)
print(example)
print('-'*10 + ' EXAMPLE END ' + '-'*10)
print(f'\nTask: {task}')
for step in range(1, MAX_STEPS + 1):
    best_overall_score = -np.inf
    # query Planning LM for single-step action candidates
    samples, log_probs = generator(curr_prompt + f'\nStep {step}:', sampling_params)
    #print(samples)
    for sample, log_prob in zip(samples, log_probs):
        most_similar_idx, matching_score = find_most_similar(sample, action_list_embedding)

        if log_prob is None:
            log_prob = 0.0
        # rank each sample by its similarity score and likelihood score
        overall_score = matching_score + BETA * log_prob
        translated_action = action_list[most_similar_idx]
        # heuristic for penalizing generating the same action as the last action
        if step > 1 and translated_action == previous_action:
            overall_score -= 0.5
        # find the translated action with highest overall score
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_action = translated_action

    # terminate early when either the following is true:
    # 1. top P*100% of samples are all 0-length (ranked by log prob)
    # 2. overall score is below CUTOFF_THRESHOLD
    # else: autoregressive generation based on previously translated action
    top_samples_ids = np.argsort(log_probs)[-int(P * len(samples)):]
    are_zero_length = all([len(samples[i]) == 0 for i in top_samples_ids])
    below_threshold = best_overall_score < CUTOFF_THRESHOLD
    if are_zero_length:
        print(f'\n[Terminating early because top {P*100}% of samples are all 0-length]')
        break
    elif below_threshold:
        print(f'\n[Terminating early because best overall score is lower than CUTOFF_THRESHOLD ({best_overall_score} < {CUTOFF_THRESHOLD})]')
        #Stepの蓋然性のスコア化
        #閾値を下回った場合、自然言語で回答を生成
        prompt_extra="次の命令に対し、全てのStepに対する評価を1から10までの整数で評価した値だけを出力してください"
        #samples, log_probs = generator(curr_prompt + f'\nStep {step}:', sampling_params)
        samples, log_probs = generator(prompt_extra+curr_prompt + f'\nStep {step}:', sampling_params)
        print(samples)
        break
    else:
        previous_action = best_action
        formatted_action = (best_action[0].upper() + best_action[1:]).replace('_', ' ') # 'open_fridge' -> 'Open fridge'
        curr_prompt += f'\nStep {step}: {formatted_action}'
        print(f'Step {step}: {formatted_action}')



