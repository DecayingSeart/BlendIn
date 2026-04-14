
from tqdm import tqdm
import os
import argparse
import json
import concurrent.futures   # for parallel processing of the samples

from openai import OpenAI

from utils import apply_instruct_template, completion_with_nudging, completion_with_baseline
from dataset_utils import extract_ans, parse_pred_ans, get_dataset, PROMPTS

import numpy as np 



def _is_correct_for_early_stop(dataset_name: str, all_info: dict) -> bool:
    """Extract correctness signal for early stopping."""
    import re

    if dataset_name in ('truthfulqa', 'xstest'):
        try:
            return int(all_info['scores'].get('correct', 0)) == 1
        except (TypeError, ValueError, AttributeError):
            return False

    elif dataset_name == 'gsm8k':
        # Rule-based: last number in pred == last number in gold
        def find_numbers(text):
            nums = re.findall(r'\d+(?:,\d{3})*(?:\.\d*)?', text)
            return [str(float(n.replace(',', ''))) for n in nums]

        pred_nums = find_numbers(all_info.get('extracted_answer', ''))
        gold_nums = find_numbers(all_info.get('gold_answer', ''))
        if pred_nums and gold_nums:
            return pred_nums[-1] == gold_nums[-1]
        return False

    return False  # unknown dataset



# ============= CAPABILITY GAP CALCULATION =============

# Known MMLU scores (approximate - update with your actual data)
MODEL_MMLU_SCORES = {
    'Llama-3.1-8B': 0.59,
    'Llama-3.2-1B-Instruct': 0.47,
    'Llama-3.2-3B-Instruct': 0.56,
    'Llama-3.2-3B': 0.53, 
    'gemma-2-9b': 0.2800,
    'gemma-3-4b-it': 0.5900,
    'gemma-3-4b-pt':  0.1500 ,
    'gemma-3-1b-it': 0.4200,
    'Qwen3-8B-Base': 0.5700,
    'Qwen3-1.7B': 0.3300,
    'Qwen3-4B': 0.5600 ,
    'Qwen3-4B-Base': 0.4300 ,
}

def get_model_capability(model_name: str) -> float:
    """Extract model capability from name."""
    model_key = model_name.split('/')[-1]
    return MODEL_MMLU_SCORES.get(model_key, 0.45)  # Default to 0.45 if unknown


def calculate_capability_gap(base_model: str, nudging_model: str) -> float:
    """
    Calculate capability gap (base - nudging).
    Positive gap = base is stronger.
    """
    base_cap = get_model_capability(base_model)
    nudging_cap = get_model_capability(nudging_model)
    return base_cap - nudging_cap


def adjust_threshold_adaptive(base_threshold: float, capability_gap: float) -> float:
    """
    Adjust intervention threshold based on capability gap.
    
    Larger gap → higher threshold → fewer interventions
    Formula: adjusted = base * (1 + gap * sensitivity)
    
    Args:
        base_threshold: Original threshold (e.g., 0.3)
        capability_gap: base_capability - nudging_capability
    
    Returns:
        Adjusted threshold (capped at 0.6)
    """
    sensitivity = 0.5  # How much gap affects threshold
    adjusted = base_threshold * (1 + capability_gap * sensitivity)
    return min(adjusted, 0.6)  # Cap at 0.6 to keep some intervention



def exp_nudging(
    client_base: OpenAI,
    client_nudging: OpenAI,
    dataset_name: str,
    num_samples: int,
    base_model: str,
    nudging_model: str,
    max_token_total: int,
    input_data: list,
    output_data: list,
    base_temperature: float,
    nudging_temperature: float,
    base_top_p: float,
    answer_start_prompt_base: str = None,
    answer_start_prompt_nudging: str = None,
    print_intermediate_output: bool = False,
    rerun: bool = False,
    exp_prefix: str = "",
    exp: str = "nudging",
    completion_token_num = 16,
    completion_token_num_nudging = 16,
    top_prob_thres: float = 0.3,
    num_threads: int = 10,

    max_intervention_rate=1,
    agreement_top_k: int = 5,                    # ← ADD
    enable_agreement_filter: bool = False,       # ← ADD

    min_nudging_confidence: float = 0.0,

    enable_distribution_blending: bool = False,    # ← ADD
    blend_alpha: float = 0.5,                      # ← ADD (or 'auto')

    verify_overlap: bool = False,

    early_stop_threshold: float = None,   # ← ADD
    early_stop_min_samples: int = 200,
):
    print("="*20)
    print(f"{exp} experiments")
    print("experiment settings:")
    print(f"dataset_name: {dataset_name}")
    print(f"num_samples: {num_samples}")
    print(f"base_model: {base_model}")
    print(f"nudging_model: {nudging_model}")
    print(f"max_token_total: {max_token_total}")
    print(f"base_temperature: {base_temperature}")
    print(f"base_top_p: {base_top_p}")
    print(f"nudging_temperature: {nudging_temperature}")
    print(f"num_threads: {num_threads}")
    print(f"top probability threshold: {top_prob_thres}")
    print(f"completion_token_num_base: {completion_token_num}")
    print(f"completion_token_num_nudging: {completion_token_num_nudging}")
    print("="*20)

    base_dir = f'./outputs/{dataset_name}'                      # for saving the txt file that contains the questions and answers
    os.makedirs(base_dir, exist_ok=True)

    all_info_base_dir = f'./outputs/{dataset_name}/all_info'    # for saving the json file that contains all the information
    os.makedirs(all_info_base_dir, exist_ok=True)

    base_model_name = base_model.split('/')[-1]
    nudging_model_name = nudging_model.split('/')[-1]

    if len(exp_prefix)>0 and exp_prefix[-1] != '_':
        exp_prefix += '_'
    save_filename = exp_prefix + f'top_prob_{top_prob_thres}_thres_{exp}_{base_model_name}_{nudging_model_name}_{num_samples}_samples.txt'
    
    save_path = os.path.join(base_dir, save_filename)
    save_path_all_info = os.path.join(all_info_base_dir, save_filename.replace('.txt', '.json'))
    all_info_list = [None] * len(input_data)

    def process_nudging_sample(client_base, client_nudging, base_model, nudging_model, system_prompt_base, system_prompt_nudging, question_prompt, answer_start_prompt_base, answer_start_prompt_nudging, max_token_total, base_temperature, base_top_p, nudging_temperature, print_intermediate_output, top_prob_thres, q, a, dataset_name):
        question = q['input']
        context = q['context']
        all_info = completion_with_nudging(
            base_model=base_model,
            client_base=client_base,
            client_nudging=client_nudging,
            nudging_model=nudging_model,
            system_prompt_base=system_prompt_base,
            system_prompt_nudging=system_prompt_nudging,
            question=question,
            context=context,
            question_prompt=question_prompt,
            answer_start_prompt_base=answer_start_prompt_base,
            answer_start_prompt_nudging=answer_start_prompt_nudging,
            completion_token_num=completion_token_num,
            completion_token_num_nudging=completion_token_num_nudging,
            max_token_total=max_token_total,
            base_temperature=base_temperature,
            top_p=base_top_p,
            nudging_temperature=nudging_temperature,
            print_intermediate_output=print_intermediate_output,
            top_prob_thres=top_prob_thres,

            max_intervention_rate=max_intervention_rate,
            agreement_top_k=agreement_top_k,              # ← ADD
            enable_agreement_filter=enable_agreement_filter,  # ← ADD

            min_nudging_confidence=min_nudging_confidence,

            enable_distribution_blending=enable_distribution_blending,
            blend_alpha=blend_alpha,

            verify_overlap=verify_overlap,

        )

        raw_answer = all_info['raw_answer']
        ans_, scores = extract_ans(dataset_name, raw_answer, input=question, question_start=question_prompt, ans_gold=a)

        all_info['extracted_answer'] = ans_ # Currectly we don't process the answer from the model, so the extracted answer is the same as the raw answer. 
        all_info['scores'] = scores         # GPT-4 scores, if any, for the extracted/raw answer.
        all_info["gold_answer"] = a
        all_info['q_prefix'] = question_prompt
        return all_info

    def process_nudging_chunk(progress_bar, client_base, client_nudging, base_model, nudging_model, system_prompt_base, system_prompt_nudging, question_prompt, answer_start_prompt_base, answer_start_prompt_nudging, max_token_total, base_temperature, base_top_p, nudging_temperature, print_intermediate_output, input_data, output_data, dataset_name):
        chunk_results = []
        for q, a in zip(input_data, output_data):
            all_info = process_nudging_sample(
                client_base=client_base,
                client_nudging=client_nudging,
                base_model=base_model,
                nudging_model=nudging_model,
                system_prompt_base=system_prompt_base,
                system_prompt_nudging=system_prompt_nudging,
                question_prompt=question_prompt,
                answer_start_prompt_base=answer_start_prompt_base,
                answer_start_prompt_nudging=answer_start_prompt_nudging,
                max_token_total=max_token_total,
                base_temperature=base_temperature,
                base_top_p=base_top_p,
                nudging_temperature=nudging_temperature,
                print_intermediate_output=print_intermediate_output,
                top_prob_thres=top_prob_thres,
                q=q, a=a, dataset_name=dataset_name
            )
            chunk_results.append(all_info)
            progress_bar.update(1)
        return chunk_results

    if os.path.exists(save_path) and not rerun:
        print(f"Load saved results from {save_path}")
    else:
        # delete file
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(save_path_all_info):
            os.remove(save_path_all_info)
        
        system_prompt_base = PROMPTS[dataset_name]['system']
        system_prompt_nudging = PROMPTS[dataset_name]['system_nudging']
        if answer_start_prompt_base is None:
            answer_start_prompt_base = PROMPTS[dataset_name]["answer_start"]
        if answer_start_prompt_nudging is None:
            answer_start_prompt_nudging = PROMPTS[dataset_name]["answer_start"]
        question_prompt = PROMPTS[dataset_name]['question']

        print(f"system_prompt_base: {system_prompt_base}")
        print(f'system_prompt_nudging: {system_prompt_nudging}')
        print(f"question_prompt: {question_prompt}")
        print(f"answer_start_prompt_base: {answer_start_prompt_base}")
        print(f"answer_start_prompt_nudging: {answer_start_prompt_nudging}")


        if early_stop_threshold is not None:
            print(f"\n[PLATEAU STOP] Halts when performance degrades after peaking above threshold={early_stop_threshold}, min_samples={early_stop_min_samples}, batch_size={num_threads}")
            completed = []
            correct_count = 0
            peak_acc = 0.0
            margin = 0.02

            # Track last known good state
            last_good_completed = []
            last_good_correct_count = 0

            for batch_start in range(0, len(input_data), num_threads):
                batch_q = input_data[batch_start:batch_start + num_threads]
                batch_a = output_data[batch_start:batch_start + num_threads]

                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures_batch = [
                        executor.submit(
                            process_nudging_sample,
                            client_base, client_nudging, base_model, nudging_model,
                            system_prompt_base, system_prompt_nudging, question_prompt,
                            answer_start_prompt_base, answer_start_prompt_nudging,
                            max_token_total, base_temperature, base_top_p, nudging_temperature,
                            print_intermediate_output, top_prob_thres, q, a, dataset_name
                        )
                        for q, a in zip(batch_q, batch_a)
                    ]
                    batch_results = [f.result() for f in futures_batch]

                completed.extend(batch_results)
                correct_count += sum(_is_correct_for_early_stop(dataset_name, info) for info in batch_results)
                running_acc = correct_count / len(completed)
                peak_acc = max(peak_acc, running_acc)
                print(f"[PLATEAU STOP] n={len(completed)} acc={running_acc:.4f} peak={peak_acc:.4f}")

                # Save state whenever acc is above threshold
                if running_acc >= early_stop_threshold + margin:
                    last_good_completed = list(completed)
                    last_good_correct_count = correct_count

                if (len(completed) >= early_stop_min_samples
                        and peak_acc >= early_stop_threshold + margin
                        and running_acc <= early_stop_threshold):
                    # Restore to last known good state
                    completed = last_good_completed
                    final_acc = last_good_correct_count / len(completed) if completed else 0
                    print(f"[PLATEAU STOP] Performance degraded after peak — restoring to peak checkpoint. Final n={len(completed)}, acc={final_acc:.4f}")
                    break

            all_info_list = completed
        else:


            chunk_size = (len(input_data) + num_threads - 1) // num_threads
            progress_bar = tqdm(total=len(input_data), desc="Processing samples")
            futures = []
            # Process the samples in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                for i in range(0, len(input_data), chunk_size):
                    chunk_input_data = input_data[i:i + chunk_size]
                    chunk_output_data = output_data[i:i + chunk_size]
                    futures.append((executor.submit(
                        process_nudging_chunk, progress_bar, client_base, client_nudging, base_model, nudging_model, 
                        system_prompt_base, system_prompt_nudging, question_prompt, answer_start_prompt_base, answer_start_prompt_nudging, 
                        max_token_total, base_temperature, base_top_p, nudging_temperature, print_intermediate_output, 
                        chunk_input_data, chunk_output_data, dataset_name), i))

                for future, start_index in futures:
                    result = future.result()
                    all_info_list[start_index:start_index + len(result)] = result

            progress_bar.close()

        # save all information
        with open(save_path_all_info, 'w') as f:
            json.dump(all_info_list, f)

        # save the answers and scores
        with open(save_path, 'a') as fd:
            for info in all_info_list:
                fd.write('Input_q: %s\nNudging_words:\n%s\nA_model:\n%s\nScores:\n%s\nA:\n%s\n\n' % (info['context'] + info['question'], info['all_nudging_words'], info['extracted_answer'], json.dumps(info['scores'], indent=4), info['gold_answer']))

    return parse_pred_ans(dataset_name, save_path, print_aggregated_metric=True)

def exp_baseline(
    client_base: OpenAI,
    client_proxy_chat: OpenAI,
    client_proxy_base: OpenAI,
    dataset_name: str,
    num_samples: int,
    base_model: str,
    proxy_chat_model: str,
    proxy_base_model: str,
    max_token_total: int,
    baseline_method: str,
    input_data: list,
    output_data: list,
    rerun: bool = False,
    exp_prefix: str = "",
    temperature: float = 0.0,
    num_threads: int = 100,
):
    print("+"*20)
    print("Baseline experiments")
    print("experiment settings:")
    print(f"baseline_method: {baseline_method}")
    print(f"dataset_name: {dataset_name}")
    print(f"num_samples: {num_samples}")
    print(f"base_model: {base_model}")
    print(f"proxy_chat_model: {proxy_chat_model}")
    print(f"proxy_base_model: {proxy_base_model}")
    print(f"max_token_total: {max_token_total}")
    print(f"temperature: {temperature}")
    print(f"num_threads: {num_threads}")
    print("+"*20)
    base_dir = f'./outputs/{dataset_name}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    base_model_name = base_model.split('/')[-1]
    proxy_chat_model_name = proxy_chat_model.split('/')[-1]
    if len(exp_prefix)>0 and exp_prefix[-1] != '_':
        exp_prefix += '_'
    save_filename = exp_prefix + f'{baseline_method}_{base_model_name}_{proxy_chat_model_name}_{num_samples}_samples.txt'
    save_path = os.path.join(base_dir, save_filename)
    all_info_base_dir = f'./outputs/{dataset_name}/all_info'
    os.makedirs(all_info_base_dir, exist_ok=True)
    save_path_all_info = os.path.join(all_info_base_dir, save_filename.replace('.txt', '.json'))
    all_info_list = [None] * len(input_data)

    def process_sample(client_base, 
                       client_proxy_chat,
                       client_proxy_base,
                       base_model, 
                       proxy_chat_model,
                       proxy_base_model,
                       baseline_method,
                       max_token_total, 
                       instruction_prompt, 
                       q_prefix, 
                       answer_start_prompt,
                       temperature, 
                       q, a, dataset_name):
        """Function for process a single sample"""
        context = q['context']
        question = q['input']

        all_info = completion_with_baseline(
            client_base=client_base,
            client_proxy_chat=client_proxy_chat,
            client_proxy_base=client_proxy_base,
            base_model=base_model,
            proxy_chat_model=proxy_chat_model,
            proxy_base_model=proxy_base_model,
            baseline_method=baseline_method,
            max_token_total=max_token_total,
            instruction_prompt=instruction_prompt,
            q_prefix=q_prefix,
            answer_start_prompt=answer_start_prompt,
            temperature=temperature,
            context=context,
            question=question,
        )
        raw_answer = all_info['raw_answer']
        ans_, scores = extract_ans(dataset_name, raw_answer, input=question, question_start=q_prefix, ans_gold=a)
        all_info['extracted_answer'] = ans_
        all_info['scores'] = scores
        all_info["gold_answer"] = a
        all_info['q_prefix'] = q_prefix

        return all_info
    
    def process_chunk(progress_bar, client_base, client_proxy_chat, client_proxy_base, base_model, proxy_chat_model, proxy_base_model, baseline_method, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, input_data, output_data, dataset_name):
        chunk_results = []
        for q, a in zip(input_data, output_data):
            all_info = process_sample(client_base=client_base, 
                                        client_proxy_chat=client_proxy_chat,
                                        client_proxy_base=client_proxy_base,
                                        base_model=base_model,
                                        proxy_chat_model=proxy_chat_model,
                                        proxy_base_model=proxy_base_model,
                                        baseline_method=baseline_method,
                                        max_token_total=max_token_total,
                                        instruction_prompt=instruction_prompt,
                                        q_prefix=q_prefix,
                                        answer_start_prompt=answer_start_prompt,
                                        temperature=temperature,
                                        q=q, a=a, dataset_name=dataset_name)
            chunk_results.append(all_info)
            progress_bar.update(1)
        return chunk_results
    
    if os.path.exists(save_path) and not rerun:
        print(f"Load saved results from {save_path}")
    else:
        # delete file
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(save_path_all_info):
            os.remove(save_path_all_info)
        instruction_prompt = PROMPTS[dataset_name]['system']
        answer_start_prompt = PROMPTS[dataset_name]["answer_start"]
        q_prefix = PROMPTS[dataset_name]['question']
        print(f"instruction_prompt: {instruction_prompt}")
        print(f"q_prefix: {q_prefix}")
        print(f"answer_start_prompt: {answer_start_prompt}")

        chunk_size = len(input_data) // num_threads
        progress_bar = tqdm(total=len(input_data), desc="Processing samples")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(input_data), chunk_size):
                chunk_input_data = input_data[i:i + chunk_size]
                chunk_output_data = output_data[i:i + chunk_size]
                futures.append((executor.submit(
                    process_chunk, progress_bar, client_base, client_proxy_chat, client_proxy_base, base_model, proxy_chat_model, proxy_base_model, baseline_method, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, chunk_input_data, chunk_output_data, dataset_name), i))

            for future, start_index in futures:
                result = future.result()
                all_info_list[start_index:start_index + len(result)] = result

        # Save results
        with open(save_path, 'a') as fd:
            for info in all_info_list:
                fd.write('Input_q: %s\nA_model:\n%s\nScores:\n%s\nA:\n%s\n\n' % (info['context'] + info['question'], info['extracted_answer'], json.dumps(info['scores'], indent=4), info['gold_answer']))

        with open(save_path_all_info, 'w') as f:
            json.dump(all_info_list, f)
        # Close the progress bar
        progress_bar.close()
    return parse_pred_ans(dataset_name, save_path, print_aggregated_metric=True)

def exp_single_model(
    client: OpenAI,
    dataset_name: str,
    num_samples: int,
    model: str,
    max_token_total: int,
    input_data: list,
    output_data: list,
    rerun: bool = False,
    exp_prefix: str = "",
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_threads: int = 100,
    model_type: str = "nudging",
):
    print("*"*20)
    print(f"{model_type} only experiments")
    print("experiment settings:")
    print(f"dataset_name: {dataset_name}")
    print(f"num_samples: {num_samples}")
    print(f"{model_type}_model: {model}")
    print(f"max_token_total: {max_token_total}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"num_threads: {num_threads}")
    print("*"*20)

    base_dir = f'./outputs/{dataset_name}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    model_name = model.split('/')[-1]
    if len(exp_prefix)>0 and exp_prefix[-1] != '_':
        exp_prefix += '_'
    filename = exp_prefix + f'{model_name}_{num_samples}_samples.txt'
    save_path = os.path.join(base_dir, filename)
    all_info_base_dir = f'./outputs/{dataset_name}/all_info'
    os.makedirs(all_info_base_dir, exist_ok=True)
    save_path_all_info = os.path.join(all_info_base_dir, filename.replace('.txt', '.json'))
    all_info_list = [None] * len(input_data)

    def process_sample(client, model, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, q, a, dataset_name):
        all_info = {}
        prompt_q = q['context'] + q_prefix + q['input']

        # apply the instruction template for the instruct models, for base models the function concatenates the prompts with "\n"
        prompt = apply_instruct_template(model_name=model, system_prompt=instruction_prompt, instruct_prompt=prompt_q, response_prompt=answer_start_prompt) 
        
        response = client.completions.create(
            model=model,
            max_tokens=max_token_total,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
        )
        ans_model = response.choices[0].text

        all_info['raw_answer'] = ans_model
        ans_, scores = extract_ans(dataset_name, ans_model, input=q['input'], question_start=q_prefix, ans_gold=a)

        all_info['question'] = q['input']
        all_info['context'] = q['context']
        all_info['q_prefix'] = q_prefix
        all_info['prompted_question'] = prompt
        all_info['system_prompt_nudging'] = instruction_prompt
        all_info['full_prompt'] = prompt
        all_info['answer_start_prompt'] = answer_start_prompt
        all_info['extracted_answer'] = ans_
        all_info['scores'] = scores
        all_info["gold_answer"] = a

        return all_info

    def process_chunk(progress_bar, client, model, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, input_data, output_data, dataset_name):
        chunk_results = []
        for q, a in zip(input_data, output_data):
            all_info = process_sample(client, model, max_token_total, instruction_prompt, q_prefix, answer_start_prompt, temperature, q, a, dataset_name)
            chunk_results.append(all_info)
            progress_bar.update(1)
        return chunk_results
    
    if os.path.exists(save_path) and not rerun:
        print(f"Load saved results from {save_path}")
    else:
        # delete file
        if os.path.exists(save_path):
            os.remove(save_path)

        instruction_prompt = PROMPTS[dataset_name]['system_nudging'] if model_type == 'nudging' else PROMPTS[dataset_name]['system']
        answer_start_prompt = PROMPTS[dataset_name]["answer_start"]
        q_prefix = PROMPTS[dataset_name]['question']
        print(f"instruction_prompt: {instruction_prompt}")
        print(f"q_prefix: {q_prefix}")
        print(f"answer_start_prompt: {answer_start_prompt}")

        chunk_size = len(input_data) // num_threads
        progress_bar = tqdm(total=len(input_data), desc="Processing samples")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(input_data), chunk_size):
                chunk_input_data = input_data[i:i + chunk_size]
                chunk_output_data = output_data[i:i + chunk_size]
                futures.append((executor.submit(
                    process_chunk, progress_bar, client, model, max_token_total, instruction_prompt, q_prefix,
                    answer_start_prompt, temperature, chunk_input_data, chunk_output_data, dataset_name), i))

            for future, start_index in futures:
                result = future.result()
                all_info_list[start_index:start_index + len(result)] = result

        # Save results
        with open(save_path, 'a') as fd:
            for info in all_info_list:
                fd.write('Input_q: %s\nA_model:\n%s\nScores:\n%s\nA:\n%s\n\n' % (info['context'] + info['question'], info['extracted_answer'], json.dumps(info['scores'], indent=4), info['gold_answer']))

        with open(save_path_all_info, 'w') as f:
            json.dump(all_info_list, f)
        # Close the progress bar
        progress_bar.close()
    return parse_pred_ans(dataset_name, save_path, print_aggregated_metric=True)

def main(
    dataset_name: str,
    num_samples: int,
    num_threads: int,
    base_model: str,
    nudging_model: str,
    proxy_base_model: str,      # for the baseline methods (proxy tuning)
    proxy_chat_model: str,   # for the baseline methods (proxy tuning)
    max_token_total: int,
    base_temperature: float,
    nudging_temperature: float,
    base_top_p: float,
    nudging_top_p: float,
    print_intermediate_output: bool,
    exp: str,
    exp_prefix: str,
    split: str,
    rerun: bool,
    baseline_method: str,
    top_prob_thres: float,
    completion_token_num: int,
    completion_token_num_nudging: int,
    base_host: str = None,
    nudging_host: str = None,
    proxy_base_host: str = None,
    proxy_chat_host: str = None,
    use_local_host: bool = True,

    adaptive_threshold: bool = False,  # ← ADD THIS
    max_intervention_rate: float = 1.0,  # ← ADD THIS

    agreement_top_k: int = 5,
    enable_agreement_filter: bool = False,

    min_nudging_confidence: float = 0.0,

    enable_distribution_blending: bool = False,    # ← ADD
    blend_alpha: float = 0.5,                      # ← ADD

    verify_overlap: bool = False,

    early_stop_threshold: float = None,
    early_stop_min_samples: int = 200,
):
    exp_prefix = f"split_{split}_" + exp_prefix
    ############################
    # For deploying the model using API providers like Together AI or Fireworks AI
    # Load the API keys
    # with open('togetherai-key.txt', 'r') as f:
    #     togetherai_api_key = f.read().strip()
    # with open('fireworks-key.txt', 'r') as f:
    #     fireworks_key = f.read().strip()
    # client_together_ai = OpenAI(
    #     api_key=togetherai_api_key,
    #     base_url="https://api.together.xyz/v1",
    # )
    # client_fireworks = OpenAI(
    #     base_url = "https://api.fireworks.ai/inference/v1",
    #     api_key=fireworks_key,
    # )
    ############################

    input_data, output_data, input_key, output_key = get_dataset(
        dataset_name=dataset_name,
        split=split,
        num_sample=num_samples
    )

    # print one example
    print("\nSample example: ")
    if input_key is not None:
        print(f"{input_key}:\n"+ input_data[0]['input'])
    else:
        print(input_data[0]['input'])
    if output_key is not None:
        print(f"{output_key}:\n"+ output_data[0])
    else:
        print(output_data[0])
    
    # set up the clients for the base model
    if use_local_host:   # local server
        openai_api_key = "EMPTY"
        client_base = OpenAI(
            api_key=openai_api_key,
            base_url=base_host,
        )
    ############################
    # Change here for deploying the model using API providers like Together AI or Fireworks AI
    # else:
    #     if base_model.startswith('accounts'):
    #         client_base = client_fireworks
    #     else:
    #         client_base = client_together_ai
    ############################

    # set up the clients for the nudging model
    if use_local_host:   # local server
        openai_api_key = "EMPTY"
        client_nudging = OpenAI(
            api_key=openai_api_key,
            base_url=nudging_host,
        )
    ############################
    # Change here for deploying the model using API providers like Together AI or Fireworks AI
    # else:
    #     if nudging_model.startswith('accounts'):
    #         client_nudging = client_fireworks
    #     else:
    #         client_nudging = client_together_ai
    ############################
    
    # set up the clients for the proxy models
    if proxy_base_model is not None:
        if use_local_host:   # local server
            openai_api_key = "EMPTY"
            client_proxy_base = OpenAI(
                api_key=openai_api_key,
                base_url=proxy_base_host,
            )
            client_proxy_chat = OpenAI(
                api_key=openai_api_key,
                base_url=proxy_chat_host,
            )
        ############################
        # Change here for deploying the model using API providers like Together AI or Fireworks AI
        # else:
        #     if proxy_base_model.startswith('accounts'):
        #         client_proxy_base = client_fireworks
        #         client_proxy_nudging = client_fireworks
        #     else:
        #         client_proxy_base = client_together_ai
        #         client_proxy_nudging = client_together_ai
        ############################



    # ========== ADD THIS SECTION (after dataset loading, before exp calls) ==========
    # Apply adaptive threshold if enabled
    original_threshold = top_prob_thres
    if adaptive_threshold and exp == 'nudging':
        capability_gap = calculate_capability_gap(base_model, nudging_model)
        top_prob_thres = adjust_threshold_adaptive(top_prob_thres, capability_gap)
        
        print("\n" + "="*60)
        print("ADAPTIVE THRESHOLD")
        print("="*60)
        print(f"Base model capability:    {get_model_capability(base_model):.2f}")
        print(f"Nudging model capability: {get_model_capability(nudging_model):.2f}")
        print(f"Capability gap:           {capability_gap:+.3f}")
        print(f"Original threshold:       {original_threshold:.3f}")
        print(f"Adjusted threshold:       {top_prob_thres:.3f}")
        if capability_gap > 0.15:
            print(f"Effect: Fewer interventions (large gap)")
        elif capability_gap < -0.15:
            print(f"Effect: More interventions (negative gap)")
        else:
            print(f"Effect: Moderate adjustment")
        print("="*60 + "\n")
    # ===============================================================================



    if exp == 'nudging':
        exp_nudging(
            client_base=client_base,
            client_nudging=client_nudging,
            dataset_name=dataset_name,
            num_samples=num_samples,
            base_model=base_model,
            nudging_model=nudging_model,
            max_token_total=max_token_total,
            base_temperature=base_temperature,
            nudging_temperature=nudging_temperature,
            base_top_p=base_top_p,
            input_data=input_data,
            output_data=output_data,
            print_intermediate_output=print_intermediate_output,
            rerun=rerun,
            exp_prefix=exp_prefix,
            exp=exp,
            completion_token_num=completion_token_num,
            completion_token_num_nudging=completion_token_num_nudging,
            top_prob_thres=top_prob_thres,
            num_threads=num_threads,

            max_intervention_rate=max_intervention_rate,  # ← ADD THIS
            agreement_top_k=agreement_top_k,              # ← ADD
            enable_agreement_filter=enable_agreement_filter,  # ← ADD

            min_nudging_confidence=min_nudging_confidence,  

            enable_distribution_blending=enable_distribution_blending,  # ← ADD THIS
            blend_alpha=blend_alpha,  

            verify_overlap=verify_overlap,  # Add this

            early_stop_threshold=early_stop_threshold,
            early_stop_min_samples=early_stop_min_samples,
        )


        # ========== ADD OVERLAP SUMMARY ==========
        if verify_overlap and enable_distribution_blending:
            print("\n" + "="*60)
            print("TOKEN DISTRIBUTION OVERLAP ANALYSIS")
            print("="*60)
            
            # Load the all_info JSON file to get overlap stats
            save_filename = exp_prefix + f'top_prob_{top_prob_thres}_thres_nudging_{base_model.split("/")[-1]}_{nudging_model.split("/")[-1]}_{num_samples}_samples.txt'
            all_info_path = f'./outputs/{dataset_name}/all_info/{save_filename.replace(".txt", ".json")}'
            
            with open(all_info_path, 'r') as f:
                all_info_list = json.load(f)
            
            # Aggregate overlap stats across all samples
            all_overlaps = []
            for info in all_info_list:
                if 'overlap_stats' in info and info['overlap_stats']:
                    all_overlaps.extend(info['overlap_stats'])
            
            if all_overlaps:
                avg_overlap = np.mean([s['overlap_pct'] for s in all_overlaps])
                avg_base_vocab = np.mean([s['base_tokens'] for s in all_overlaps])
                avg_nudging_vocab = np.mean([s['nudging_tokens'] for s in all_overlaps])
                avg_common = np.mean([s['common_tokens'] for s in all_overlaps])
                
                print(f"Interventions analyzed: {len(all_overlaps)}")
                print(f"Average base vocab size: {avg_base_vocab:.1f} tokens")
                print(f"Average nudging vocab size: {avg_nudging_vocab:.1f} tokens")
                print(f"Average common tokens: {avg_common:.1f}")
                print(f"Average overlap: {avg_overlap:.1f}%")
                
                if avg_overlap < 20:
                    print("⚠️  LOW OVERLAP - Models have very different vocabularies")
                elif avg_overlap < 50:
                    print("ℹ️  MODERATE OVERLAP - Some vocabulary sharing")
                else:
                    print("✓  HIGH OVERLAP - Models share vocabulary well")
            else:
                print("No overlap data collected (base model always confident)")
            
            print("="*60 + "\n")
        # ==========================================


    elif exp == 'base_only':
        exp_single_model(
            client=client_base,
            dataset_name=dataset_name,
            num_samples=num_samples,
            model=base_model,
            max_token_total=max_token_total,
            input_data=input_data,
            output_data=output_data,
            rerun=rerun,
            exp_prefix=exp_prefix,
            num_threads=num_threads,
            temperature=base_temperature,
            top_p=base_top_p,
            model_type="base",
        )
    elif exp == 'nudging_only':
        exp_single_model(
            client=client_nudging,
            dataset_name=dataset_name,
            num_samples=num_samples,
            model=nudging_model,
            max_token_total=max_token_total,
            input_data=input_data,
            output_data=output_data,
            rerun=rerun,
            exp_prefix=exp_prefix,
            num_threads=num_threads,
            temperature=nudging_temperature,
            top_p=nudging_top_p,
            model_type="nudging",
        )
    elif exp == 'baseline':
        exp_baseline(
            client_base=client_base,
            client_proxy_chat=client_proxy_chat,
            client_proxy_base=client_proxy_base,
            dataset_name=dataset_name,
            num_samples=num_samples,
            base_model=base_model,
            proxy_base_model=proxy_base_model,
            proxy_chat_model=proxy_chat_model,
            max_token_total=max_token_total,
            baseline_method=baseline_method,
            temperature=base_temperature,
            input_data=input_data,
            output_data=output_data,
            rerun=rerun,
            exp_prefix=exp_prefix,
            num_threads=num_threads,
        )
    else:
        raise ValueError(f"Unknown experiment {exp}")


    # Calculate capability gap if adaptive threshold is enabled
#    if adaptive_threshold and exp == 'nudging':
#        capability_gap = calculate_capability_gap(base_model, nudging_model)
#        original_threshold = top_prob_thres
#        top_prob_thres = adjust_threshold_for_capability_gap(top_prob_thres, capability_gap)
#        
#        print(f"\n[ADAPTIVE THRESHOLD]")
#        print(f"Capability gap (base - nudging MMLU): {capability_gap:.3f}")
#        print(f"Original threshold: {original_threshold:.3f}")
#        print(f"Adjusted threshold: {top_prob_thres:.3f}")
#        print(f"Expected effect: {'Less intervention (larger gap)' if capability_gap > 0.15 else 'Moderate adjustment'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="The name of the dataset")
    parser.add_argument("--num_samples", type=int, default=None, help="The number of samples")
    parser.add_argument("--num_threads", type=int, default=20, help="The number of threads")
    parser.add_argument("--base_model", type=str, default=None, help="The base model to use")
    parser.add_argument("--nudging_model", type=str, default=None, help="The nudging model to use")
    parser.add_argument("--proxy_base_model", type=str, default=None, help="The proxy base model to use")
    parser.add_argument("--proxy_chat_model", type=str, default=None, help="The proxy chat model to use")
    parser.add_argument("--completion_token_num", type=int, default=16, help="The number of token to complete in each nudging round using the base model")
    parser.add_argument("--completion_token_num_nudging", type=int, default=16, help="The number of token to complete in each nudging round using the nudging model")
    parser.add_argument("--max_token_total", type=int, default=512, help="The maximum number of tokens")
    parser.add_argument("--base_temperature", type=float, default=0.0, help="The temperature for the base model")
    parser.add_argument("--base_top_p", type=float, default=0.9, help="The top p for the base model")       # by default, we have temperature = 0 so top_p is not used by default
    parser.add_argument("--nudging_temperature", type=float, default=0.0, help="The temperature for the nudging model")
    parser.add_argument("--nudging_top_p", type=float, default=0.9, help="The top p for the nudging model") # by default, we have temperature = 0 so top_p is not used by default
    parser.add_argument("--baseline_method", type=str, choices=["ensemble", 'proxy_tuning'], default='ensemble', help="The baseline method name")
    parser.add_argument("--top_prob_thres", type=float, default=0.3, help="The top-1 token probability threshold for top prob nudging")
    parser.add_argument("--exp", type=str, choices=['nudging_only', "base_only", 'nudging', 'baseline'], default='nudging_only', help="The experiment to run")
    parser.add_argument("--exp_prefix", type=str, default="", help="The prefix for the experiment, e.g. complex_fewshot_")
    parser.add_argument("--split", type=str, default='test', help="The split to test")
    parser.add_argument("--base_host", type=str, default=None, help="The base host for local models")
    parser.add_argument("--nudging_host", type=str, default=None, help="The nudging host for local models")
    parser.add_argument("--proxy_base_host", type=str, default=None, help="The proxy base host for local models")
    parser.add_argument("--proxy_chat_host", type=str, default=None, help="The proxy chat host for local models")
    parser.add_argument("--rerun", action='store_true', help="Whether to rerun the experiment")
    parser.add_argument("--print_intermediate_output", action='store_true', help="Whether to print intermediate output")

    parser.add_argument("--adaptive_threshold", action='store_true', 
                    help="Adjust intervention threshold based on capability gap")
    parser.add_argument("--max_intervention_rate", type=float, default=0.15,
                    help="Maximum intervention rate (default 0.15 = 15%)")
    parser.add_argument("--agreement_top_k", type=int, default=5,
                       help="Check if nudging in base model's top-k (default 5)")
    parser.add_argument("--enable_agreement_filter", action='store_true',
                       help="Enable agreement-based filtering")
    parser.add_argument("--min_nudging_confidence", type=float, default=0.0,    # ← ADD
                       help="Minimum nudging confidence (0.0-1.0, 0.0 = disabled)")
    parser.add_argument("--enable_distribution_blending", action='store_true',
                       help="Enable true distribution blending")
    parser.add_argument("--blend_alpha", type=str, default="0.5",
                       help="Blend weight (0.0-1.0 or 'auto')")

    parser.add_argument("--verify_overlap", action='store_true',
                       help="Print debug information during generation")

    parser.add_argument("--early_stop_threshold", type=float, default=None,
                        help="Halt when performance degrades after peaking above this value; restores to peak checkpoint")
    parser.add_argument("--early_stop_min_samples", type=int, default=200,
                        help="Minimum samples before plateau detection is active (default 200)")

    args = parser.parse_args()


    # Convert blend_alpha
    if args.blend_alpha == 'auto':
        blend_alpha = 'auto'
    else:
        blend_alpha = float(args.blend_alpha)


    args = vars(args)


    args['blend_alpha'] = blend_alpha  # ← Actually update the dict!


    main(**args)


