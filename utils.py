import numpy as np

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") # use gpt3.5 tokenizer for token number controlling, so we don't need to load the actual tokenizer for API models

NUM_LOGPROBS = {
    'top_prob': 1,
}




def blend_distributions(base_logprobs, nudging_logprobs, alpha=0.5):
    """
    Blend two probability distributions.
    
    Args:
        base_logprobs: dict {token: logprob} from base model
        nudging_logprobs: dict {token: logprob} from nudging model
        alpha: float in [0,1], weight for nudging (0=base only, 1=nudging only)
    
    Returns:
        dict {token: blended_logprob}
    """
    import numpy as np
    
    # Convert logprobs to probs
    base_probs = {k: np.exp(v) for k, v in base_logprobs.items()}
    nudging_probs = {k: np.exp(v) for k, v in nudging_logprobs.items()}
    
    # Get all tokens from both distributions
    all_tokens = set(base_probs.keys()) | set(nudging_probs.keys())
    
    # Blend probabilities
    blended_probs = {}
    for token in all_tokens:
        base_p = base_probs.get(token, 0.0)
        nudging_p = nudging_probs.get(token, 0.0)
        blended_probs[token] = alpha * nudging_p + (1 - alpha) * base_p
    
    # Normalize (should already be normalized, but ensure)
    total = sum(blended_probs.values())
    if total > 0:
        blended_probs = {k: v/total for k, v in blended_probs.items()}
    
    # Convert back to logprobs
    blended_logprobs = {k: np.log(v + 1e-10) for k, v in blended_probs.items()}
    
    return blended_logprobs

def sample_from_logprobs(logprobs, temperature=0.0):
    """
    Sample token from logprob distribution.
    
    Args:
        logprobs: dict {token: logprob}
        temperature: float, 0.0 for greedy (deterministic)
    
    Returns:
        sampled token (str)
    """
    import numpy as np
    
    if temperature == 0.0:
        # Greedy: return token with highest logprob
        return max(logprobs.items(), key=lambda x: x[1])[0]
    else:
        # Sample with temperature
        tokens = list(logprobs.keys())
        logprobs_array = np.array([logprobs[t] for t in tokens])
        
        # Apply temperature
        logprobs_array = logprobs_array / temperature
        
        # Convert to probabilities
        probs = np.exp(logprobs_array)
        probs = probs / np.sum(probs)
        
        # Sample
        sampled_idx = np.random.choice(len(tokens), p=probs)
        return tokens[sampled_idx]

def compute_blend_alpha_old(base_logprobs, nudging_logprobs, base_uncertain_threshold=0.3):
    """
    Compute blend weight alpha based on base model's uncertainty.
    
    Higher alpha = trust nudging more
    Lower alpha = trust base more
    
    Args:
        base_logprobs: dict {token: logprob} from base
        nudging_logprobs: dict {token: logprob} from nudging  
        base_uncertain_threshold: threshold for base uncertainty
    
    Returns:
        alpha: float in [0, 1]
    """
    import numpy as np
    
    # Get base model's max probability
    base_probs = {k: np.exp(v) for k, v in base_logprobs.items()}
    base_max_prob = max(base_probs.values())
    
    # Compute alpha based on base uncertainty
    if base_max_prob < base_uncertain_threshold:
        # Base very uncertain → trust nudging more
        alpha = 0.7
    elif base_max_prob < base_uncertain_threshold + 0.2:
        # Base moderately uncertain → moderate blend
        alpha = 0.5
    else:
        # Base confident → trust base more
        alpha = 0.3
    
    return alpha


def compute_blend_alpha_noextreme(base_logprobs, nudging_logprobs, tau=0.4):
    import numpy as np
    
    base_probs = {k: np.exp(v) for k, v in base_logprobs.items()}
    nudging_probs = {k: np.exp(v) for k, v in nudging_logprobs.items()}
    
    base_max_prob = max(base_probs.values())
    nudging_max_prob = max(nudging_probs.values())
    
    nudging_top = max(nudging_probs, key=nudging_probs.get)
    agreement = base_probs.get(nudging_top, 0.0)
    
    # Continuous: high when base uncertain AND nudging confident AND agree
    base_uncertainty = 1 - base_max_prob
    alpha_raw = base_uncertainty * nudging_max_prob * (1 + agreement)
    
    # Scale to [0.1, 0.8]
    alpha = 0.1 + alpha_raw * 0.7
    return float(np.clip(alpha, 0.1, 0.9))

def compute_blend_alpha_biased(base_logprobs, nudging_logprobs, tau=0.4):
    import numpy as np
    
    base_probs = {k: np.exp(v) for k, v in base_logprobs.items()}
    nudging_probs = {k: np.exp(v) for k, v in nudging_logprobs.items()}
    
    base_max_prob = max(base_probs.values())
    nudging_max_prob = max(nudging_probs.values())
    
    nudging_top = max(nudging_probs, key=nudging_probs.get)
    agreement = base_probs.get(nudging_top, 0.0)
    
    # Natural range: all signals in [0,1], product in [0,2]
    base_uncertainty = 1 - base_max_prob
    alpha = base_uncertainty * nudging_max_prob * (1 + agreement)
    
    # Clip to (0,1) only for validity
    return float(np.clip(alpha, 0.0, 1.0))


def compute_blend_alpha(base_logprobs, nudging_logprobs, tau=0.4):
    base_probs = {k: np.exp(v) for k, v in base_logprobs.items()}
    nudging_probs = {k: np.exp(v) for k, v in nudging_logprobs.items()}
    
    base_max_prob = max(base_probs.values())
    nudging_max_prob = max(nudging_probs.values())
    
    # Base alpha from relative confidence - naturally ∈ [0,1]
    alpha = nudging_max_prob / (base_max_prob + nudging_max_prob)
    
    # Agreement bonus: if nudging's top token appears in base distribution
    nudging_top = max(nudging_probs, key=nudging_probs.get)
    agreement = base_probs.get(nudging_top, 0.0)
    
    # Small agreement bonus, keep bounded
    # Max bonus = 0.1 when agreement = 1.0
    alpha = alpha + 0.1 * agreement
    
    return float(np.clip(alpha, 0.0, 1.0))



def apply_instruct_template(model_name, system_prompt, instruct_prompt, response_prompt, add_bos=False):
    model_name = model_name.lower()

    if "chat" in model_name and "llama" in model_name and "2" in model_name:
        return llama_2_chat_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos)
    elif "instruct" in model_name and "llama" in model_name and "3" in model_name:
        if "3.1" in model_name: # for llama-3.1 models, add knowledge cut in system prompmt
            return llama_3_instruct_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos, add_knowledge_cut=True)
        else:
            return llama_3_instruct_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos)
    elif "it" in model_name and "gemma" in model_name:
        return gemma_instruct_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos)
    elif "instruct" in model_name and "olmo" in model_name:
        return olmo_instruct_template(system_prompt=system_prompt, instruct_prompt=instruct_prompt, response_prompt=response_prompt, add_bos=add_bos) 
    else:
        return f"{system_prompt}\n{instruct_prompt}\n{response_prompt}" # non-instruct model or models with unknown template

def llama_2_chat_template(system_prompt, instruct_prompt, response_prompt, add_bos=False):
    """
    Convert the input and output into the template used for the llama-2 chat models training.
    """
    prefix = "<s>" if add_bos else ""
    return prefix + f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruct_prompt} [/INST] {response_prompt.lstrip()}"  # for most servers that add <s> automatically so we don't need to add it here

def llama_3_instruct_template(system_prompt, instruct_prompt, response_prompt, add_bos=False, add_knowledge_cut=False):
    """
    Convert the input and output into the template used for the llama-3 instruct models training.
    """
    # print("applying llama-3 instruct template")
    prefix = "<|begin_of_text|>" if add_bos else ""
    if add_knowledge_cut:
        system_prompt = f"Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"+ system_prompt
    return prefix + f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruct_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response_prompt}"

def gemma_instruct_template(system_prompt, instruct_prompt, response_prompt, add_bos=False):
    """
    Convert the input and output into the template used for the gemma instruct models training.
    <bos><start_of_turn>user
    Write a hello world program<end_of_turn>
    <start_of_turn>model
    """
    prefix = "<bos>" if add_bos else ""
    return prefix + f"<start_of_turn>user\n{system_prompt}\n{instruct_prompt}<end_of_turn>\n<start_of_turn>model\n{response_prompt}"

def olmo_instruct_template(system_prompt, instruct_prompt, response_prompt, add_bos=False):
    """
    Convert the input and output into the template used for the olmo instruct models training.
    """
    return f"<|endoftext|><|user|>\n{system_prompt}\n{instruct_prompt}\n<|assistant|>\n{response_prompt}"

def find_longest_repeated_suffix(s):
    
    # Helper function to check if a substring repeats
    def has_repeated(s, length):
        if length < 30:
            return False
        # Extract the suffix of length 'length'
        suffix = s[-length:]
        # Check the rest of the string for another occurrence
        # return s[:-length].find(suffix) != -1
        return s[:-length].endswith(suffix)

    left, right = 0, len(s)
    result = 0

    # Binary search for the longest repeated suffix
    while left <= right:
        mid = (left + right) // 2
        if has_repeated(s, mid):
            result = mid  # Store the longest length found
            left = mid + 1  # Try for a longer suffix
        else:
            right = mid - 1  # Try for a shorter suffix

    # Return the longest repeated suffix
    if result > 0:
        return s[-result:]
    return None  # Return an empty string if no repetition is found

def remove_redundant_repetitions(s):
    s = s.strip()
    # Find the longest repeated suffix
    longest_repeated_suffix = find_longest_repeated_suffix(s)
    while longest_repeated_suffix:
        # Remove the longest repeated suffix
        s = s[:-len(longest_repeated_suffix)]
        # Find the longest repeated suffix again
        longest_repeated_suffix = find_longest_repeated_suffix(s)
    return s

def repetition_check(new_completion, full_prefix, subseq_len=5):
    words = new_completion.split(" ")
    if len(words) > subseq_len and new_completion in full_prefix:
        return True
    return False

def check_need_nudging(nudging_method,
                        base_token_id,
                        current_base_info, 
                        thresholds,
):
    if nudging_method == 'top_prob':
        # check if the token prob is below the threshold
        sorted_base_top_logprobs = {k: v for k, v in sorted(current_base_info["top_logprobs"][base_token_id].items(), key=lambda item: item[1], reverse=True)}
        base_top_prob = np.exp(list(sorted_base_top_logprobs.values())[0])
        need_nudging = base_top_prob < thresholds['top_prob']
    else:
        raise ValueError(f"Unknown nudging method {nudging_method}")
    return need_nudging

def complete_with_base(nudging_method='top_prob',
                        base_model="davinci-002",
                        full_prefix_base="",
                        output="",
                        current_base_info=None,
                        max_completion_token=256,
                        completion_token_num=16,
                        client_base=None,
                        thresholds=None,
                        temperature=0.0,
                        top_p=0.9,
                        ):
    completion_base = "" if len(current_base_info["completion"]) == 0 else current_base_info["tokens"][0]   # accept the first token from the 1st round which is the acc token from the first stage
    completion_all = "" if len(current_base_info["completion"]) == 0 else current_base_info["tokens"][0]    # completion_all records all the tokens from the base model including the tokens that are not accepted in the last round, for debugging and visualization
    found_nudging_token = False
    response = None
    has_acc_token_stage_1 = True if len(current_base_info["completion"]) > 0 else False                     # if the current_base_info["completion"] is not empty, it means the first token in base completion is accepted from the 1st stage
    EMPTY_INFO_DICT = {
        "completion": "",
        "tokens": [],
        "top_logprobs": [],
        "stop_reason": None, 
        "num_logprobs": NUM_LOGPROBS[nudging_method],
    }
    next_nudging_info = EMPTY_INFO_DICT     # for nudging methods that compute nudging info during base completion, we can save the info for the next round, currently not used for top_prob nudging
    while len(encoding.encode(completion_base, disallowed_special=())) < max_completion_token and not found_nudging_token:
       
        if current_base_info["completion"] == "":
            # complete the sentence using the base model
            response = client_base.completions.create(
                model=base_model,
                prompt=full_prefix_base + output + completion_base,
                max_tokens=completion_token_num,
                temperature=temperature,
                logprobs=current_base_info["num_logprobs"],
                top_p=top_p,
                )
            current_base_info["tokens"] = response.choices[0].logprobs.tokens
            current_base_info["top_logprobs"] = response.choices[0].logprobs.top_logprobs
            current_base_info["completion"] = response.choices[0].text

        if has_acc_token_stage_1:
            # pop the first token from the 1st round as it is already accepted from stage 1
            current_base_info["tokens"] = current_base_info["tokens"][1:]
            current_base_info["top_logprobs"] = current_base_info["top_logprobs"][1:]
            current_base_info["completion"] = "".join(current_base_info["tokens"])
            has_acc_token_stage_1 = False

        completion = current_base_info["completion"]
        tokens = current_base_info["tokens"]

        if completion in completion_base:
            break   # repeated completion, break

        nudging_position = -1

        # find the first token that violates the nudging criteria
        for base_idx in range(len(tokens)):
            found_nudging_token = check_need_nudging(nudging_method=nudging_method, base_token_id=base_idx, current_base_info=current_base_info, thresholds=thresholds)
            if found_nudging_token:
                nudging_position = base_idx
                break
        
        if nudging_position == -1:
            new_completion= "".join(tokens)
        else:
            new_completion = "".join(tokens[:nudging_position])   # include the last agreed token
        # avoid repetition in answer
        if repetition_check(new_completion, output + completion_base):
            break
        else:
            completion_base += new_completion

        if found_nudging_token: # if found the nudging token, break the loop, concat the last base completion to completion_all
            completion_all += completion
        else:
            completion_all += new_completion

        next_nudging_info = EMPTY_INFO_DICT
        if response is not None and response.choices[0].finish_reason == "stop":
            break

        # reset the current_base_info
        current_base_info['completion'] = ""
        current_base_info['tokens'] = []
        current_base_info['top_logprobs'] = []

    return completion_base, completion_all, next_nudging_info

def completion_with_nudging(
        base_model="davinci-002",
        nudging_model="gpt-3.5-turbo",
        system_prompt_base="Answer the question by walking through the reasoning step by step.",
        system_prompt_nudging="Answer the question by walking through the reasoning step by step.",
        question="",
        context="",
        question_prompt="Question: ",
        answer_start_prompt_base="Answer: ",
        answer_start_prompt_nudging="Answer: ",
        completion_token_num=16,
        completion_token_num_nudging=16,
        max_token_total=256,
        print_intermediate_output=False,
        client=None,                # default client
        client_base=None,
        client_nudging=None,
        max_round=150,
        nudging_temperature=0.0,    # deterministic for nudging
        base_temperature=0.0,       # deterministic for base model
        nudging_method='top_prob',
        top_prob_thres=0.3,
        top_p=0.9,

        max_intervention_rate=1.0,  # ← ADD THIS (1.0 = disabled, 0.15 = 15% cap)
        agreement_top_k=5,                   # ← ADD: Check top-k tokens
        enable_agreement_filter=False,       # ← ADD: Enable/disable filter

        min_nudging_confidence=0.0,

        enable_distribution_blending=False,    # ← NEW
        blend_alpha=0.5,                       # ← NEW (or 'auto' for adaptive)

        verify_overlap=False
        ):
    if client_base is None:
        client_base = client
    if client_nudging is None:
        client_nudging = client

    if nudging_method not in NUM_LOGPROBS.keys():
        raise ValueError(f"nudging method {nudging_method} number of logprobs not defined")

    full_prefix_base = apply_instruct_template(base_model, system_prompt_base, context + question_prompt + question, answer_start_prompt_base)  # for base model this function just adds newlines
    full_prefix_nudging = apply_instruct_template(nudging_model, system_prompt_nudging, context + question_prompt + question, answer_start_prompt_nudging)

    thresholds = {
        'top_prob': top_prob_thres,
    }


    # ========== ADD THESE LINES ==========
    # Intervention rate tracking
    total_tokens_generated = 0
    intervention_count = 0
    intervention_capped = False

    # Agreement filter tracking
    rejected_nudging_count = 0
    accepted_nudging_count = 0
    # =====================================


    output = ""


    token_count = 0
    intervention_count = 0
    blended_token_count = 0
    
    stop_reason = None




    # ===================================================================
    # BLENDING APPROACH: Token-by-token generation with distribution blending
    # ===================================================================
    
    if enable_distribution_blending:
        overlap_stats = []  # ← ADD THIS at start of blending loop
        while len(encoding.encode(output, disallowed_special=())) < max_token_total: #and token_count < max_round:
            token_count += 1
            
            # Get base model's distribution
            response_base = client_base.completions.create(
                model=base_model,
                prompt=full_prefix_base + output,
                max_tokens=1,
                temperature=base_temperature,
                logprobs=100,  # Get full distribution (API max is usually 5-100)
                top_p=top_p,
            )
            base_logprobs = response_base.choices[0].logprobs.top_logprobs[0]
            base_top_token = response_base.choices[0].logprobs.tokens[0]
            
            # Check if base is uncertain
            base_probs = {k: np.exp(v) for k, v in base_logprobs.items()}
            base_max_prob = max(base_probs.values())
            base_uncertain = base_max_prob < top_prob_thres
            
            if base_uncertain:
                # Base uncertain → get nudging distribution and blend
                intervention_count += 1
                
                response_nudging = client_nudging.completions.create(
                    model=nudging_model,
                    prompt=full_prefix_nudging + output,
                    max_tokens=1,
                    temperature=nudging_temperature,
                    logprobs=100,
                )
                nudging_logprobs = response_nudging.choices[0].logprobs.top_logprobs[0]
                
                # DEBUG: Check if flag is set
                #print(f"[DEBUG] verify_overlap={verify_overlap}, intervention_count={intervention_count}")
                # ========== ADD DEBUG CODE HERE ==========
                # Check token overlap (only print on first intervention for each sample)
                #if verify_overlap:
                #    common_tokens = set(base_logprobs.keys()) & set(nudging_logprobs.keys())
                #    print(f"[OVERLAP CHECK] Common tokens: {len(common_tokens)}/{len(base_logprobs)} base, {len(common_tokens)}/{len(nudging_logprobs)} nudging")
                #    print(f"[OVERLAP CHECK] Sample base tokens: {list(base_logprobs.keys())[:5]}")
                #    print(f"[OVERLAP CHECK] Sample nudging tokens: {list(nudging_logprobs.keys())[:5]}")
                # ==========================================
                # ========== ADD OVERLAP TRACKING ==========
                if verify_overlap:
                    common_tokens = set(base_logprobs.keys()) & set(nudging_logprobs.keys())
                    overlap_stats.append({
                        'position': token_count,
                        'base_tokens': len(base_logprobs),
                        'nudging_tokens': len(nudging_logprobs),
                        'common_tokens': len(common_tokens),
                        'overlap_pct': 100 * len(common_tokens) / len(base_logprobs),
                    })
                # ==========================================



                # Compute blend weight (adaptive or fixed)
                if blend_alpha == 'auto':
                    alpha = compute_blend_alpha(base_logprobs, nudging_logprobs, top_prob_thres)
                else:
                    alpha = blend_alpha
                
                # Blend distributions
                blended_logprobs = blend_distributions(base_logprobs, nudging_logprobs, alpha)
                
                # Sample from blended distribution
                next_token = sample_from_logprobs(blended_logprobs, temperature=0.0)  # Greedy
                blended_token_count += 1
                
                if print_intermediate_output:
                    nudging_top = max(nudging_logprobs.items(), key=lambda x: x[1])[0]
                    print(f"[BLEND α={alpha:.2f}] Base: '{base_top_token}' ({base_max_prob:.3f}), "
                          f"Nudging: '{nudging_top}', Selected: '{next_token}'")
            else:
                # Base confident → use base only
                next_token = base_top_token
                
                if print_intermediate_output:
                    print(f"[BASE] Token: '{next_token}' (prob={base_max_prob:.3f})")
            
            # Add token to output
            output += next_token
            
            # Check for stop
            if response_base.choices[0].finish_reason == "stop":
                stop_reason = "base_model_stop"
                break
        
        if token_count >= max_round and not stop_reason:
            stop_reason = "round"
        if len(encoding.encode(output, disallowed_special=())) >= max_token_total and not stop_reason:
            stop_reason = "length"
        
        all_info = {
            "question": question,
            "context": context,
            "raw_answer": output,
            "all_nudging_words": [],  # Not applicable for blending
            "all_completions": [output],
            "stop_reason": stop_reason,
            "system_prompt_base": system_prompt_base,
            "system_prompt_nudging": system_prompt_nudging,
            "full_prefix_base": full_prefix_base,
            "full_prefix_nudging": full_prefix_nudging,
            "intervention_count": intervention_count,
            "blended_token_count": blended_token_count,
            "total_tokens": token_count,
            "intervention_rate": intervention_count / max(token_count, 1),

            "overlap_stats": overlap_stats if verify_overlap else [],  # ← ADD THIS

        }
        
        return all_info
    
    # ===================================================================
    # ORIGINAL NUDGING APPROACH (if blending disabled)
    # ===================================================================

    else:


        nudging_round = 0
        all_nudging_words = []
        all_nudging_and_completions = []
        current_nudging_info = {
            "completion": "",
            "tokens": [],
            "top_logprobs": [],
            "stop_reason": None,
            "num_logprobs": NUM_LOGPROBS[nudging_method],
        }
        stop_reason = None
        repeat_nudging_word = 0
        last_nudging_word = ""
        while len(encoding.encode(output, disallowed_special=())) < max_token_total and nudging_round < max_round:    # use the number of gpt-3.5 token to approximately control the length
            nudging_round += 1
            if current_nudging_info["completion"] == "":
                response = client_nudging.completions.create(
                    model=nudging_model,
                    prompt=full_prefix_nudging + output,
                    max_tokens=completion_token_num_nudging,
                    temperature=nudging_temperature,
                    logprobs=current_nudging_info["num_logprobs"],
                    )
                current_nudging_info["completion"] = response.choices[0].text
                current_nudging_info["tokens"] = response.choices[0].logprobs.tokens
                current_nudging_info["top_logprobs"] = response.choices[0].logprobs.top_logprobs
                current_nudging_info["stop_reason"] = response.choices[0].finish_reason

            # if finish_reason is stop, break the loop, also handles nudging completion from previous round
            if current_nudging_info["stop_reason"] == "stop":
                stop_reason = "nudging_model_stop"
                if len(current_nudging_info["completion"]) > 0:
                    all_nudging_words.append(current_nudging_info["completion"])
                    all_nudging_and_completions.append(current_nudging_info["completion"])
                    output += current_nudging_info["completion"]
                break


            # ========== ADD THESE LINES ==========
            # Check if intervention budget exceeded
            current_rate = intervention_count / total_tokens_generated if total_tokens_generated > 20 else 0
            skip_nudging_this_round = (current_rate >= max_intervention_rate) and (total_tokens_generated > 20)
            

            # ========== ADD THESE LINES ==========
            # If intervention budget exceeded, skip nudging and use base model only
            if skip_nudging_this_round:
                #print(f"[CAP TRIGGERED] Round {nudging_round}")
                intervention_capped = True
                # Just use base model completion without nudging guidance
                response = client_base.completions.create(
                    model=base_model,
                    prompt=full_prefix_base + output,
                    max_tokens=completion_token_num,
                    temperature=base_temperature,
                    top_p=top_p,
                )
                base_completion = response.choices[0].text
                output += base_completion
                total_tokens_generated += len(response.choices[0].logprobs.tokens) if response.choices[0].logprobs else 1
                
                # Reset nudging info for next round
                current_nudging_info = {
                    "completion": "",
                    "tokens": [],
                    "top_logprobs": [],
                    "stop_reason": None,
                    "num_logprobs": NUM_LOGPROBS[nudging_method],
                }
                continue
            # =====================================





            # ===================================================================
            # Stage 1: use base model to find the first token that violates the nudging criteria (no need to nudge)
            # ===================================================================
            found_acc_token = False
            current_base_info = {   # will be passed to the next stage
                "completion": "",
                "tokens": [],
                "top_logprobs": [],
                "num_logprobs": NUM_LOGPROBS[nudging_method],
            }
            nudging_text = current_nudging_info["completion"]
            num_whitespaces = len(nudging_text) - len(nudging_text.lstrip(" "))
            space_prefix = " " * num_whitespaces
            current_nudging_words = nudging_text.lstrip(" ").split(" ")     # token leads to some unexpected behaviors, still use nudging word
            nudging_word_id = 0 if len(current_nudging_words) > 1 else 1    # if only one word, always accept the word and go to the next round: it won't go into the loop and found_acc_token will be False
            while not found_acc_token and nudging_word_id < len(current_nudging_words) - 1:
                nudging_word_id += 1                # always accept the first word
                nudging_gen_prefix = space_prefix + " ".join(current_nudging_words[:nudging_word_id])
                current_nudging_word = " " + current_nudging_words[nudging_word_id]  # add a leading space to the current nudging word since the nudging words a split by space
                if current_nudging_word == " ":     # skip the multiple space
                    continue
                prefix = full_prefix_base + output + nudging_gen_prefix
                response = client_base.completions.create(
                    model=base_model,
                    prompt=prefix,
                    max_tokens=completion_token_num,
                    temperature=base_temperature,
                    logprobs=current_base_info["num_logprobs"],
                    top_p=top_p,
                    )
                current_base_info["tokens"] = response.choices[0].logprobs.tokens
                current_base_info["top_logprobs"] = response.choices[0].logprobs.top_logprobs
                current_base_info["completion"] = response.choices[0].text

                # look for the first token that meets the nudging criteria
                first_base_token = current_base_info["tokens"][0]            
                if current_nudging_word.startswith(first_base_token): # check if the current nudging word is the same or starts with the first base token
                    found_acc_token = True
                else: 
                    found_acc_token = not check_need_nudging(nudging_method,    # check if the token violates the nudging criteria (no need to nudge)
                                                            base_token_id=0,
                                                            current_base_info=current_base_info, 
                                                            thresholds=thresholds)
                    
            # here we have either prefix_idx == len(current_nudging_info["tokens"]):    if no token meets the nudging criteria, use the current nudging completion
            # or found_acc_token == True:    if a token violates the nudging criteria, we use the prefix as nudging tokens
            
            nudging_words = space_prefix +  " ".join(current_nudging_words[:nudging_word_id])
            


#            # ========== AGREEMENT FILTER START ==========
#            if enable_agreement_filter and len(nudging_words.strip()) > 0:
#                # Get base model's top-k preferences at current position
#                check_prefix = full_prefix_base + output
#                response_check = client_base.completions.create(
#                    model=base_model,
#                    prompt=check_prefix,
#                    max_tokens=1,
#                    temperature=base_temperature,
#                    logprobs=agreement_top_k,
#                    top_p=top_p,
#                )
                
#                base_top_logprobs = response_check.choices[0].logprobs.top_logprobs[0]
#                base_top_tokens = [t.strip() for t in base_top_logprobs.keys()]
                
                # Get first word of nudging suggestion
#                first_nudging_word = nudging_words.strip().split()[0] if nudging_words.strip() else ""
                
                # Check if nudging aligns with any of base model's top-k choices
#                nudging_agrees = False
#                for base_token in base_top_tokens:
#                    if (first_nudging_word.startswith(base_token) or 
#                        base_token.startswith(first_nudging_word) or
#                        first_nudging_word == base_token):
#                        nudging_agrees = True
#                        break
                
#                if not nudging_agrees:
                    # Base model strongly disagrees - reject this nudging
#                    rejected_nudging_count += 1
#                    if print_intermediate_output:
#                        print(f"[REJECTED] Nudging '{first_nudging_word}' not in base top-{agreement_top_k}: {base_top_tokens}")
                    
                    # Reset and skip to next round
#                    current_nudging_info = {
#                        "completion": "",
#                        "tokens": [],
#                        "logprobs": [],
#                        "stop_reason": None,
#                        "num_logprobs": NUM_LOGPROBS[nudging_method],
#                    }
#                    continue
#                else:
#                    accepted_nudging_count += 1
            # ========== AGREEMENT FILTER END ==========



    # ========== DUAL FILTER: Agreement + Confidence ==========
    #        if enable_agreement_filter and len(nudging_words.strip()) > 0:
    #            # Get base model's top-k preferences at current position
    #            check_prefix = full_prefix_base + output
    #            response_check = client_base.completions.create(
    #                model=base_model,
    #                prompt=check_prefix,
    #                max_tokens=1,
    #                temperature=base_temperature,
    #                logprobs=agreement_top_k,
    #                top_p=top_p,
    #            )
                
    #            base_top_logprobs = response_check.choices[0].logprobs.top_logprobs[0]
    #            base_top_tokens = [t.strip() for t in base_top_logprobs.keys()]
                
                # ========== NEW: Get nudging model's confidence ==========
    #            if min_nudging_confidence > 0.0:
                    # Query nudging model for its confidence at current position
    #                response_nudging = client_nudging.completions.create(
    #                    model=nudging_model,
    #                    prompt=full_prefix_nudging + output,
    #                    max_tokens=1,
    #                    temperature=nudging_temperature,
    #                    logprobs=1,  # Just need top-1 for confidence
    #                )
    #               nudging_logprobs = response_nudging.choices[0].logprobs.top_logprobs[0]
                    
                    # Get confidence for nudging's suggestion
    #                first_nudging_word = nudging_words.strip().split()[0]
                    
                    # Find the token in nudging's distribution
    #                nudging_confidence = 0.0
    #                for token, logprob in nudging_logprobs.items():
    #                    if (first_nudging_word.startswith(token.strip()) or 
    #                        token.strip().startswith(first_nudging_word) or
    #                        first_nudging_word == token.strip()):
    #                        nudging_confidence = np.exp(logprob)
    #                        break
    #            else:
    #                nudging_confidence = 1.0  # Confidence check disabled
                # ========================================================
                
                # Get first word of nudging suggestion
    #            first_nudging_word = nudging_words.strip().split()[0] if nudging_words.strip() else ""
                
                # ========== DUAL FILTER CHECK ==========
                # Check 1: Agreement
    #            nudging_agrees = False
    #            for base_token in base_top_tokens:
    #                if (first_nudging_word.startswith(base_token) or 
    #                    base_token.startswith(first_nudging_word) or
    #                    first_nudging_word == base_token):
    #                    nudging_agrees = True
    #                    break
                
                # Check 2: Confidence
    #            nudging_confident = nudging_confidence >= min_nudging_confidence
                
                # Decision: BOTH conditions must be satisfied
    #            if not (nudging_agrees and nudging_confident):
                    # Reject: Base model strongly disagrees OR nudging uncertain
    #                rejected_nudging_count += 1
    #                if print_intermediate_output:
    #                    print(f"[REJECTED] Word: '{first_nudging_word}', "
    #                          f"Agreement: {nudging_agrees}, "
    #                          f"Confidence: {nudging_confidence:.3f}, "
    #                          f"Base top-{agreement_top_k}: {base_top_tokens}")
                    
                    # Reset and skip to next round
    #                current_nudging_info = {
    #                    "completion": "",
    #                    "tokens": [],
    #                    "logprobs": [],
    #                    "stop_reason": None,
    #                    "num_logprobs": NUM_LOGPROBS[nudging_method],
    #                }
    #                continue
    #            else:
    #                accepted_nudging_count += 1
    #                if print_intermediate_output:
    #                    print(f"[ACCEPTED] Word: '{first_nudging_word}', "
    #                          f"Confidence: {nudging_confidence:.3f}")
            # ========== END DUAL FILTER ==========

    # ========== CONFIDENCE COMPETITION ==========
            if enable_agreement_filter and len(nudging_words.strip()) > 0:
                # Get base model's top-k preferences at current position
                check_prefix = full_prefix_base + output
                response_check = client_base.completions.create(
                    model=base_model,
                    prompt=check_prefix,
                    max_tokens=1,
                    temperature=base_temperature,
                    logprobs=agreement_top_k,  # Get top-k for agreement check
                    top_p=top_p,
                )
                
                base_top_logprobs = response_check.choices[0].logprobs.top_logprobs[0]
                base_top_tokens = [t.strip() for t in base_top_logprobs.keys()]  # ← DEFINE THIS
                
                # Get first word of nudging suggestion
                first_nudging_word = nudging_words.strip().split()[0] if nudging_words.strip() else ""
                
                # Check 1: Agreement (safety gate)
                nudging_agrees = False
                for base_token in base_top_tokens:
                    if (first_nudging_word.startswith(base_token) or 
                        base_token.startswith(first_nudging_word) or
                        first_nudging_word == base_token):
                        nudging_agrees = True
                        break
                
                if not nudging_agrees:
                    # Safety: reject incoherent suggestions
                    rejected_nudging_count += 1
                    if print_intermediate_output:
                        print(f"[REJECTED - INCOHERENT] Word: '{first_nudging_word}', Base top-{agreement_top_k}: {base_top_tokens}")
                    
                    current_nudging_info = {
                        "completion": "",
                        "tokens": [],
                        "logprobs": [],
                        "stop_reason": None,
                        "num_logprobs": NUM_LOGPROBS[nudging_method],
                    }
                    continue
                
                # Check 2: Confidence competition
                # Get base's confidence for its top choice
                base_confidence = np.exp(list(base_top_logprobs.values())[0])
                
                # Get nudging's confidence at current position
                response_nudging = client_nudging.completions.create(
                    model=nudging_model,
                    prompt=full_prefix_nudging + output,
                    max_tokens=1,
                    temperature=nudging_temperature,
                    logprobs=5,  # Get top-5 to find our token
                )
                nudging_top_logprobs = response_nudging.choices[0].logprobs.top_logprobs[0]
                
                # Find nudging's confidence for its suggested token
                nudging_confidence = 0.0
                for token, logprob in nudging_top_logprobs.items():
                    if (first_nudging_word.startswith(token.strip()) or 
                        token.strip().startswith(first_nudging_word) or
                        first_nudging_word == token.strip()):
                        nudging_confidence = np.exp(logprob)
                        break
                
                # If we didn't find the token in top-5, get more logprobs
                if nudging_confidence == 0.0:
                    response_nudging_full = client_nudging.completions.create(
                        model=nudging_model,
                        prompt=full_prefix_nudging + output,
                        max_tokens=1,
                        temperature=nudging_temperature,
                        logprobs=20,  # Get more tokens
                    )
                    nudging_full_logprobs = response_nudging_full.choices[0].logprobs.top_logprobs[0]
                    for token, logprob in nudging_full_logprobs.items():
                        if (first_nudging_word.startswith(token.strip()) or 
                            token.strip().startswith(first_nudging_word) or
                            first_nudging_word == token.strip()):
                            nudging_confidence = np.exp(logprob)
                            break
                
                # Competition: choose more confident
                if nudging_confidence > base_confidence:
                    # Nudging wins - more confident
                    accepted_nudging_count += 1
                    if print_intermediate_output:
                        print(f"[ACCEPTED - MORE CONFIDENT] Nudging: {nudging_confidence:.3f} > Base: {base_confidence:.3f}")
                    # Continue with nudging words (don't break)
                else:
                    # Base wins - more confident or equal
                    rejected_nudging_count += 1
                    if print_intermediate_output:
                        print(f"[REJECTED - LESS CONFIDENT] Nudging: {nudging_confidence:.3f} <= Base: {base_confidence:.3f}")
                    
                    # Reset and skip to next round (use base)
                    current_nudging_info = {
                        "completion": "",
                        "tokens": [],
                        "logprobs": [],
                        "stop_reason": None,
                        "num_logprobs": NUM_LOGPROBS[nudging_method],
                    }
                    continue
            # ========== END CONFIDENCE COMPETITION ==========


#            # ===== CONFIDENCE THRESHOLD =====
#            if min_nudging_confidence > 0.0 and len(nudging_words.strip()) > 0:
#                response_nudging_conf = client_nudging.completions.create(
#                    model=nudging_model,
#                    prompt=full_prefix_nudging + output,
#                    max_tokens=1,
#                    temperature=nudging_temperature,
#                    logprobs=1,
#                )
#                nudging_top_logprobs = response_nudging_conf.choices[0].logprobs.top_logprobs[0]
#                first_nudging_word = nudging_words.strip().split()[0] if nudging_words.strip() else ""
#                
#                nudging_confidence = 0.0
#                for token, logprob in nudging_top_logprobs.items():
#                    if (first_nudging_word.startswith(token.strip()) or
#                        token.strip().startswith(first_nudging_word) or
#                        first_nudging_word == token.strip()):
#                        nudging_confidence = np.exp(logprob)
#                        break
                
#                if nudging_confidence < min_nudging_confidence:
#                    rejected_nudging_count += 1
#                    if print_intermediate_output:
#                        print(f"[REJECTED] Nudging confidence {nudging_confidence:.3f} < threshold {min_nudging_confidence:.3f}")
#                    current_nudging_info = {
#                        "completion": "",
#                        "tokens": [],
#                        "top_logprobs": [],
#                        "stop_reason": None,
#                        "num_logprobs": NUM_LOGPROBS[nudging_method],
#                    }
#                    continue
            # ================================


            # Heuristic: if the nudging words are the same as the last one for three rounds, break the loop
            if nudging_words == last_nudging_word:
                repeat_nudging_word += 1
                if repeat_nudging_word >= 3:
                    stop_reason = "repeated_nudging_words"
                    break
            else:
                last_nudging_word = nudging_words
                repeat_nudging_word = 0
            all_nudging_words.append(nudging_words)
            output += nudging_words


            # ========== ADD THIS LINE ==========
            intervention_count += 1  # Count this as an intervention
            # ===================================



            if not found_acc_token: # if no base token can be accepted, use the current nudging completion and go to the next round
                all_nudging_and_completions.append(nudging_words)
                # reset the current nudging info and continue to the next round
                current_nudging_info = {
                    "completion": "",
                    "tokens": [],
                    "logprobs": [],
                    "stop_reason": None,
                    "num_logprobs": NUM_LOGPROBS[nudging_method],
                }
                continue
            if current_base_info["completion"] == "":   # the base model thinks the completion is done, go to the next round. Make sure current_base_info["completion"] is not empty if proceed to the next stage
                all_nudging_and_completions.append(nudging_words)
                current_nudging_info = {
                    "completion": "",
                    "tokens": [],
                    "logprobs": [],
                    "stop_reason": None,
                    "num_logprobs": NUM_LOGPROBS[nudging_method],
                }
                continue

            # ===================================================================
            # Stage 2: use nudging model to find the first token that meets the nudging criteria (need to nudge)
            # ===================================================================
            max_completion_token = max_token_total - len(encoding.encode(output, disallowed_special=()))
            completion_base, completion_base_all, current_nudging_info = complete_with_base(nudging_method=nudging_method,
                                                                                            base_model=base_model,
                                                                                            full_prefix_base=full_prefix_base,
                                                                                            output=output,
                                                                                            current_base_info=current_base_info,
                                                                                            max_completion_token=max_completion_token,
                                                                                            completion_token_num=completion_token_num,
                                                                                            client_base=client_base,
                                                                                            thresholds=thresholds,
                                                                                            temperature=base_temperature,
                                                                                            top_p=top_p,
                                                                                            )
            # print(f"next_nudging_info: {current_nudging_info}") # debug

            output += completion_base
            all_nudging_and_completions.append(nudging_words + completion_base) # the generated tokens in each round, concating all completion would be the final output


            # ========== ADD THIS LINE ==========
            total_tokens_generated += len(encoding.encode(completion_base, disallowed_special=()))
            # ===================================


            if print_intermediate_output:
                print(f"************nudging round {nudging_round}************")
                print(f"****nudging words from {nudging_model}****: {nudging_words}")
                print(f"****nudging text****: {nudging_text}")
                print(f"****completion from {base_model}****: {completion_base}")
                print(f"****all completion from {base_model}****: {completion_base_all}")
                print(f"****output****: {output}")
        
        if nudging_round >= max_round and not stop_reason:
            stop_reason = "round"
        if len(encoding.encode(output, disallowed_special=())) >= max_token_total and not stop_reason:
            stop_reason = "length"
        output = remove_redundant_repetitions(output)
        if print_intermediate_output:
            print(f"************final output************")
            print(f"****output****: {output}")

        all_info = {
            "question": question,
            "context": context,
            "raw_answer": output,
            "all_nudging_words": all_nudging_words,
            "all_completions": all_nudging_and_completions,
            "stop_reason": stop_reason,
            "system_prompt_base": system_prompt_base,
            "system_prompt_nudging": system_prompt_nudging,
            "full_prefix_base": full_prefix_base,
            "full_prefix_nudging": full_prefix_nudging,


            # ========== ADD THESE LINES ==========
            "intervention_rate": intervention_count / total_tokens_generated if total_tokens_generated > 0 else 0,
            "intervention_count": intervention_count,
            "total_tokens_generated": total_tokens_generated,
            "intervention_capped": intervention_capped,
            "rejected_nudging_count": rejected_nudging_count,
            "accepted_nudging_count": accepted_nudging_count,
            "agreement_filter_enabled": enable_agreement_filter,
            # =====================================


        }
        return all_info

############################################################################################################
# Baseline completion
############################################################################################################
def completion_baseline_ensemble(
        base_model="davinci-002",
        proxy_chat_model="gpt-3.5-turbo",
        client_base=None,
        client_proxy_chat=None,
        max_token_total=256,
        full_prefix_base="",
        full_prefix_proxy_chat="",
        temperature=0.0,
        completion_token_num=16,
        logprobs=5,
        ):
    output = ''
    while len(encoding.encode(output, disallowed_special=())) < max_token_total:
        response_base = client_base.completions.create(
            model=base_model,
            prompt=full_prefix_base + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        response_proxy_chat = client_proxy_chat.completions.create(
            model=proxy_chat_model,
            prompt=full_prefix_proxy_chat + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        base_tokens = response_base.choices[0].logprobs.tokens
        proxy_chat_tokens = response_proxy_chat.choices[0].logprobs.tokens
        base_logprobs = response_base.choices[0].logprobs.top_logprobs
        proxy_chat_logprobs = response_proxy_chat.choices[0].logprobs.top_logprobs
        # stop criteria: if chat model finish reason is stop, break the loop
        if response_proxy_chat.choices[0].finish_reason == "stop":
            output += "".join(proxy_chat_tokens)
            break
        acc_tokens = []
        for i in range(len(base_tokens)):
            if base_tokens[i] == proxy_chat_tokens[i]:
                acc_tokens.append(base_tokens[i])
            else:
                base_top_logprobs = {k: v for k, v in sorted(base_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
                proxy_chat_top_logprobs = {k: v for k, v in sorted(proxy_chat_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
                all_keys = set(base_top_logprobs.keys()).union(proxy_chat_top_logprobs.keys())
                ensemble_top_probs = {}
                for key in all_keys:
                    base_prob = np.exp(base_top_logprobs.get(key, -1e6))
                    proxy_chat_prob = np.exp(proxy_chat_top_logprobs.get(key, -1e6))
                    ensemble_top_probs[key] = base_prob + proxy_chat_prob
                ensemble_top_probs = {k: v for k, v in sorted(ensemble_top_probs.items(), key=lambda item: item[1], reverse=True)}
                acc_tokens.append(list(ensemble_top_probs.keys())[0])
                break
        new_completion = "".join(acc_tokens)
        if len(new_completion) == 0:
            # if no token is accepted, add the first non-empty token from the base model
            for i in range(len(base_tokens)):
                if len(base_tokens[i]) > 0:
                    acc_tokens.append(base_tokens[i])
                    break
            # if base model has no token, add the first token from the proxy chat model
            new_completion = "".join(acc_tokens)
            if len(new_completion) == 0:
                for i in range(len(proxy_chat_tokens)):
                    if len(proxy_chat_tokens[i]) > 0:
                        acc_tokens.append(proxy_chat_tokens[i])
                        break
        output += "".join(acc_tokens)
    return output

def completion_baseline_proxy_tuning(
        base_model="davinci-002",
        proxy_chat_model="gpt-3.5-turbo",
        proxy_base_model="davinci-002",
        client_base=None,
        client_proxy_chat=None,
        client_proxy_base=None,
        max_token_total=256,
        full_prefix_base="",
        full_prefix_proxy_chat="",
        full_prefix_proxy_base="",
        temperature=0.0,
        completion_token_num=16,
        logprobs=100,
        ):
    output = ''
    while len(encoding.encode(output, disallowed_special=())) < max_token_total:
        response_base = client_base.completions.create(
            model=base_model,
            prompt=full_prefix_base + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        response_proxy_chat = client_proxy_chat.completions.create(
            model=proxy_chat_model,
            prompt=full_prefix_proxy_chat + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        response_proxy_base = client_proxy_base.completions.create(
            model=proxy_base_model,
            prompt=full_prefix_proxy_base + output,
            max_tokens=completion_token_num,
            temperature=temperature,
            logprobs=logprobs,
            )
        base_tokens = response_base.choices[0].logprobs.tokens
        proxy_chat_tokens = response_proxy_chat.choices[0].logprobs.tokens
        proxy_base_tokens = response_proxy_base.choices[0].logprobs.tokens
        base_logprobs = response_base.choices[0].logprobs.top_logprobs
        proxy_chat_logprobs = response_proxy_chat.choices[0].logprobs.top_logprobs
        proxy_base_logprobs = response_proxy_base.choices[0].logprobs.top_logprobs
        # stop criteria: if chat model finish reason is stop, break the loop
        if response_proxy_chat.choices[0].finish_reason == "stop":
            output += "".join(proxy_chat_tokens)
            break
        acc_tokens = []
        for i in range(len(base_tokens)):
            base_top_logprobs = {k: v for k, v in sorted(base_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
            proxy_chat_top_logprobs = {k: v for k, v in sorted(proxy_chat_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
            proxy_base_top_logprobs = {k: v for k, v in sorted(proxy_base_logprobs[i].items(), key=lambda item: item[1], reverse=True)}
            shared_keys = set(base_top_logprobs.keys()).intersection(proxy_chat_top_logprobs.keys()).intersection(proxy_base_top_logprobs.keys())
            rescaled_probs = {}
            for key in shared_keys:
                base_prob = np.exp(base_top_logprobs[key])
                proxy_chat_prob = np.exp(proxy_chat_top_logprobs[key])
                proxy_base_prob = np.exp(proxy_base_top_logprobs[key])
                rescaled_probs[key] = base_prob * proxy_chat_prob / proxy_base_prob
            rescaled_probs = {k: v for k, v in sorted(rescaled_probs.items(), key=lambda item: item[1], reverse=True)}
            if len(rescaled_probs) == 0:
                acc_tokens.append(base_tokens[i])
                top_1_token = None
            else:
                top_1_token = list(rescaled_probs.keys())[0]
                acc_tokens.append(top_1_token)
            if top_1_token == base_tokens[i] and top_1_token == proxy_chat_tokens[i] and top_1_token == proxy_base_tokens[i]:   # if all models agree, continue to the next token
                continue
            else:
                break
        new_completion = "".join(acc_tokens)
        if len(new_completion) == 0:
            # if no token is accepted, add the first non-empty token from the base model
            for i in range(len(base_tokens)):
                if len(base_tokens[i]) > 0:
                    acc_tokens.append(base_tokens[i])
                    break
            # if base model has no token, add the first token from the proxy chat model
            new_completion = "".join(acc_tokens)
            if len(new_completion) == 0:
                for i in range(len(proxy_chat_tokens)):
                    if len(proxy_chat_tokens[i]) > 0:
                        acc_tokens.append(proxy_chat_tokens[i])
                        break
        output += "".join(acc_tokens)
    return output

def completion_with_baseline(client_base,
                            client_proxy_chat,
                            client_proxy_base,
                            base_model,
                            proxy_chat_model,
                            proxy_base_model,
                            baseline_method,
                            max_token_total=512,
                            instruction_prompt="",
                            q_prefix="Question: ",
                            answer_start_prompt="",
                            temperature=0,
                            context="",
                            question="",
                            completion_token_num=16,
                            ):
    all_info = {}
    full_prefix_base = apply_instruct_template(base_model, instruction_prompt, context + q_prefix + question, answer_start_prompt)
    full_prefix_proxy_chat = apply_instruct_template(proxy_chat_model, instruction_prompt, context + q_prefix + question, answer_start_prompt)
    full_prefix_proxy_base = apply_instruct_template(proxy_base_model, instruction_prompt, context + q_prefix + question, answer_start_prompt)

    if baseline_method == 'ensemble':
        ans_model = completion_baseline_ensemble(
            base_model=base_model,
            proxy_chat_model=proxy_chat_model,
            client_base=client_base,
            client_proxy_chat=client_proxy_chat,
            max_token_total=max_token_total,
            full_prefix_base=full_prefix_base,
            full_prefix_proxy_chat=full_prefix_proxy_chat,
            temperature=temperature,
            completion_token_num=completion_token_num,
        )
    elif baseline_method == 'proxy_tuning':
        ans_model = completion_baseline_proxy_tuning(
            base_model=base_model,
            proxy_chat_model=proxy_chat_model,
            proxy_base_model=proxy_base_model,
            client_base=client_base,
            client_proxy_chat=client_proxy_chat,
            client_proxy_base=client_proxy_base,
            max_token_total=max_token_total,
            full_prefix_base=full_prefix_base,
            full_prefix_proxy_chat=full_prefix_proxy_chat,
            full_prefix_proxy_base=full_prefix_proxy_base,
            temperature=temperature,
            completion_token_num=completion_token_num,
        )
    else:
        raise ValueError(f"Unknown baseline method {baseline_method}")
    all_info['raw_answer'] = ans_model
    all_info['question'] = question
    all_info['context'] = context
    all_info["full_prefix_proxy_chat"] = full_prefix_proxy_chat
    return all_info