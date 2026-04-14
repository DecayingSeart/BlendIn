import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset

import json  # ← ADD THIS
import os    # ← ADD THIS
from datetime import datetime  # ← ADD THIS

# Set Times font globally
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Times New Roman']
#plt.rcParams['mathtext.fontset'] = 'stix'  # For math symbols
#plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['ps.fonttype'] = 42

#plt.rcParams.update({
#    'font.size': 18,           # Base font size
#    'axes.titlesize': 18,      # Subplot titles
#    'axes.labelsize': 18,      # Axis labels
#    'xtick.labelsize': 18,     # X tick labels
#    'ytick.labelsize': 18,     # Y tick labels
#    'legend.fontsize': 18,     # Legend
#})



def generate_reviewer_topp_response(results):
    """
    Standalone figure + printout specifically addressing Reviewer 2's top-p suggestion.
    Produces a clean 1-panel scatter comparing top-50 vs top-p(0.9) overlap,
    plus a printed table.
    """
    from scipy.stats import pearsonr

    performance_data = {
        'gsm8k': {
            'Llama-3 1B→8B': 0.58, 'Gemma-3 1B→9B': 0.66, 'Qwen-3 1.7B→8B': 0.12,
            'Llama→Gemma': 0.67, 'Gemma→Llama': 0.59, 'Llama→Qwen': 0.87,
            'Gemma→Qwen': 0.82, 'Qwen→Llama': 0.27, 'Qwen→Gemma': 0.54,
        },
        'mmlu': {
            'Llama-3 1B→8B': 0.49, 'Gemma-3 1B→9B': 0.49, 'Qwen-3 1.7B→8B': 0.21,
            'Llama→Gemma': 0.60, 'Gemma→Llama': 0.47, 'Llama→Qwen': 0.63,
            'Gemma→Qwen': 0.70, 'Qwen→Llama': 0.41, 'Qwen→Gemma': 0.22,
        },
        'truthfulqa': {
            'Llama-3 1B→8B': 0.51, 'Gemma-3 1B→9B': 0.43, 'Qwen-3 1.7B→8B': 0.52,
            'Llama→Gemma': 0.47, 'Gemma→Llama': 0.45, 'Llama→Qwen': 0.61,
            'Gemma→Qwen': 0.59, 'Qwen→Llama': 0.48, 'Qwen→Gemma': 0.40,
        },
        'arc': {
            'Llama-3 1B→8B': 0.71, 'Gemma-3 1B→9B': 0.72, 'Qwen-3 1.7B→8B': 0.18,
            'Llama→Gemma': 0.75, 'Gemma→Llama': 0.64, 'Llama→Qwen': 0.84,
            'Gemma→Qwen': 0.86, 'Qwen→Llama': 0.46, 'Qwen→Gemma': 0.47,
        },
        'xstest': {
            'Llama-3 1B→8B': 0.12, 'Gemma-3 1B→9B': 0.10, 'Qwen-3 1.7B→8B': 0.08,
            'Llama→Gemma': 0.10, 'Gemma→Llama': 0.08, 'Llama→Qwen': 0.18,
            'Gemma→Qwen': 0.09, 'Qwen→Llama': 0.03, 'Qwen→Gemma': 0.02,
        },
        'justeval': {
            'Llama-3 1B→8B': 4.86, 'Gemma-3 1B→9B': 4.92, 'Qwen-3 1.7B→8B': 4.20,
            'Llama→Gemma': 4.82, 'Gemma→Llama': 4.89, 'Llama→Qwen': 4.93,
            'Gemma→Qwen': 4.98, 'Qwen→Llama': 3.22, 'Qwen→Gemma': 3.19,
        },
    }

    pairs = list(results.keys())

    # ---- Printed table ----
    print("\n" + "="*75)
    print("REVIEWER RESPONSE: Top-p(0.9) vs Top-50 Vocabulary Overlap")
    print("="*75)
    print(f"{'Model Pair':<25} {'Top-50 Overlap':>15} {'Top-p(0.9) Overlap':>20} {'Ratio':>8}")
    print("-"*75)
    for p in pairs:
        t50 = results[p]['mean_top_50_overlap']
        tp90 = results[p]['mean_top_p_90_overlap']
        ratio = tp90 / t50 if t50 > 0 else float('inf')
        print(f"{p:<25} {t50:>13.1f}% {tp90:>18.1f}% {ratio:>7.2f}x")
    print("-"*75)
    print("Ratio < 1.0 confirms top-p is stricter (smaller overlap sets).\n")

    # ---- Correlation table under both metrics ----
    print(f"{'Benchmark':<15} {'r (top-50)':>12} {'p (top-50)':>12} {'r (top-p0.9)':>14} {'p (top-p0.9)':>14}")
    print("-"*75)
    for benchmark in ['gsm8k', 'mmlu', 'truthfulqa', 'arc', 'xstest', 'justeval']:
        perf = [performance_data[benchmark].get(p, 0) for p in pairs]
        ov50  = [results[p]['mean_top_50_overlap'] for p in pairs]
        ovp90 = [results[p]['mean_top_p_90_overlap'] for p in pairs]

        r50, p50   = pearsonr(ov50,  perf)
        rp90, pp90 = pearsonr(ovp90, perf)
        print(f"{benchmark:<15} {r50:>12.3f} {p50:>12.3f} {rp90:>14.3f} {pp90:>14.3f}")
    print("-"*75)

    # ---- Figure: side-by-side scatter for GSM8K ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Reviewer Response: Top-p(0.9) Does Not Reveal Hidden Correlation",
                 fontsize=14, fontweight='bold')

    for ax, metric_key, label in zip(
        axes,
        ['mean_top_50_overlap', 'mean_top_p_90_overlap'],
        ['Top-50 Overlap (%)', 'Top-p(0.9) Overlap (%)']
    ):
        perf = [performance_data['gsm8k'].get(p, 0) for p in pairs]
        ov   = [results[p][metric_key] for p in pairs]
        r, pv = pearsonr(ov, perf)

        ax.scatter(ov, perf, s=100, alpha=0.8, color='#3498db')
        for i, p in enumerate(pairs):
            ax.annotate(p, (ov[i], perf[i]), fontsize=7, ha='right', alpha=0.7)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('GSM8K Performance', fontsize=12)
        ax.set_title(f'r={r:.3f}, p={pv:.3f}', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('reviewer_topp_response.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("\n✓ Figure saved to reviewer_topp_response.pdf")
    plt.show()



def generate_reviewer_topp_response(results):
    from scipy.stats import pearsonr

    performance_data = {
        'gsm8k': {
            'Llama-3 1B→8B': 0.58, 'Gemma-3 1B→9B': 0.66, 'Qwen-3 1.7B→8B': 0.12,
            'Llama→Gemma': 0.67, 'Gemma→Llama': 0.59, 'Llama→Qwen': 0.87,
            'Gemma→Qwen': 0.82, 'Qwen→Llama': 0.27, 'Qwen→Gemma': 0.54,
        },
        'mmlu': {
            'Llama-3 1B→8B': 0.49, 'Gemma-3 1B→9B': 0.49, 'Qwen-3 1.7B→8B': 0.21,
            'Llama→Gemma': 0.60, 'Gemma→Llama': 0.47, 'Llama→Qwen': 0.63,
            'Gemma→Qwen': 0.70, 'Qwen→Llama': 0.41, 'Qwen→Gemma': 0.22,
        },
        'truthfulqa': {
            'Llama-3 1B→8B': 0.51, 'Gemma-3 1B→9B': 0.43, 'Qwen-3 1.7B→8B': 0.52,
            'Llama→Gemma': 0.47, 'Gemma→Llama': 0.45, 'Llama→Qwen': 0.61,
            'Gemma→Qwen': 0.59, 'Qwen→Llama': 0.48, 'Qwen→Gemma': 0.40,
        },
        'arc': {
            'Llama-3 1B→8B': 0.71, 'Gemma-3 1B→9B': 0.72, 'Qwen-3 1.7B→8B': 0.18,
            'Llama→Gemma': 0.75, 'Gemma→Llama': 0.64, 'Llama→Qwen': 0.84,
            'Gemma→Qwen': 0.86, 'Qwen→Llama': 0.46, 'Qwen→Gemma': 0.47,
        },
        'xstest': {
            'Llama-3 1B→8B': 0.12, 'Gemma-3 1B→9B': 0.10, 'Qwen-3 1.7B→8B': 0.08,
            'Llama→Gemma': 0.10, 'Gemma→Llama': 0.08, 'Llama→Qwen': 0.18,
            'Gemma→Qwen': 0.09, 'Qwen→Llama': 0.03, 'Qwen→Gemma': 0.02,
        },
        'justeval': {
            'Llama-3 1B→8B': 4.86, 'Gemma-3 1B→9B': 4.92, 'Qwen-3 1.7B→8B': 4.20,
            'Llama→Gemma': 4.82, 'Gemma→Llama': 4.89, 'Llama→Qwen': 4.93,
            'Gemma→Qwen': 4.98, 'Qwen→Llama': 3.22, 'Qwen→Gemma': 3.19,
        },
    }

    pairs = list(results.keys())

    print("\n" + "="*75)
    print("REVIEWER RESPONSE: Top-p(0.9) vs Top-50 Overlap")
    print("="*75)
    print(f"{'Model Pair':<25} {'Top-50':>10} {'Top-p(0.9)':>12} {'Smaller?':>10}")
    print("-"*75)
    for p in pairs:
        t50  = results[p]['mean_top_50_overlap']
        tp90 = results[p]['mean_top_p_90_overlap']
        print(f"{p:<25} {t50:>8.1f}%  {tp90:>10.1f}%  {'✓' if tp90 < t50 else '✗':>10}")

    print("\n" + f"{'Benchmark':<15} {'r(top-50)':>10} {'p(top-50)':>10} {'r(top-p90)':>12} {'p(top-p90)':>12} {'Null holds?':>12}")
    print("-"*75)
    for bm in ['gsm8k', 'mmlu', 'truthfulqa', 'arc', 'xstest', 'justeval']:
        perf  = [performance_data[bm].get(p, 0) for p in pairs]
        ov50  = [results[p]['mean_top_50_overlap'] for p in pairs]
        ovp90 = [results[p]['mean_top_p_90_overlap'] for p in pairs]
        r50,  p50  = pearsonr(ov50,  perf)
        rp90, pp90 = pearsonr(ovp90, perf)
        null = '✓' if pp90 >= 0.05 else '✗'
        print(f"{bm:<15} {r50:>10.3f} {p50:>10.3f} {rp90:>12.3f} {pp90:>12.3f} {null:>12}")
    print("="*75)



def get_top_p_set(probs, p=0.90):
    """Return set of token indices covering top-p probability mass."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    # Include all tokens until cumulative mass exceeds p
    cutoff = (cumulative <= p).sum().item() + 1  # +1 to include the token that crosses threshold
    return set(sorted_indices[:cutoff].cpu().tolist())


# ============= SAVE/LOAD RESULTS ============= ← ADD THIS ENTIRE SECTION

def save_results(results, filename='vocab_analysis_results.json'):
    """Save results to JSON file."""
    # Convert any non-serializable types
    serializable_results = {}
    for pair, metrics in results.items():
        serializable_results[pair] = {
            k: float(v) if v != float('inf') else None 
            for k, v in metrics.items()
        }
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': serializable_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {filename}")


def load_results(filename='vocab_analysis_results.json'):
    """Load results from JSON file."""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert None back to inf
    results = {}
    for pair, metrics in data['results'].items():
        results[pair] = {
            k: float('inf') if v is None else v 
            for k, v in metrics.items()
        }
    
    print(f"\n✓ Results loaded from {filename}")
    print(f"  Timestamp: {data['timestamp']}")
    return results


# ============= BENCHMARK LOADING =============

def load_benchmark_prompts(dataset_name, n_samples=50):
    """
    Load prompts from various benchmarks.
    
    Args:
        dataset_name: 'gsm8k', 'mmlu', 'truthfulqa', 'arc', 'xstest'
        n_samples: Number of samples to load
    
    Returns:
        List of prompt strings
    """
    prompts = []
    
    if dataset_name == 'gsm8k':
        dataset = load_dataset('gsm8k', 'main', split='test')
        for i in range(min(n_samples, len(dataset))):
            question = dataset[i]['question']
            prompts.append(f"Solve this math problem: {question}\nAnswer:")
    
    elif dataset_name == 'mmlu':
        # Load a few subjects from MMLU
        dataset = load_dataset('cais/mmlu', 'all', split='test')
        for i in range(min(n_samples, len(dataset))):
            question = dataset[i]['question']
            choices = dataset[i]['choices']
            prompt = f"{question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
            prompts.append(prompt)
    
    elif dataset_name == 'truthfulqa':
        dataset = load_dataset('truthful_qa', 'generation', split='validation')
        for i in range(min(n_samples, len(dataset))):
            question = dataset[i]['question']
            prompts.append(f"{question}\nAnswer:")
    
    elif dataset_name == 'arc':
        dataset = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
        for i in range(min(n_samples, len(dataset))):
            question = dataset[i]['question']
            choices = dataset[i]['choices']['text']
            labels = dataset[i]['choices']['label']
            prompt = f"{question}\n"
            for label, choice in zip(labels, choices):
                prompt += f"{label}) {choice}\n"
            prompt += "Answer:"
            prompts.append(prompt)
    
    elif dataset_name == 'xstest':
        # XSTest - you need to provide how you access this
        # Option 1: If you have local file
        dataset = load_dataset("walledai/XSTest", split='test') #load_from_disk('/path/to/xstest')
        
        # Option 2: If from HuggingFace
        #dataset = load_dataset('xstest', split='test')  # ← FILL IN CORRECT PATH
        for i in range(min(n_samples, len(dataset))):
            # Adjust field name based on your dataset structure
            prompts.append(dataset[i]['prompt'])  # ← VERIFY FIELD NAME
    
    elif dataset_name == 'justeval':
        # JustEval-Safe - you need to provide how you access this
        # Option 1: If you have local file
        # dataset = load_from_disk('/path/to/justeval')
        
        # Option 2: If from HuggingFace  
        justeval = load_dataset('re-align/just-eval-instruct', 'judgements_safety')#, split='test')  # ← FILL IN CORRECT PATH
        dataset = justeval['gpt_3.5_turbo_0301']
        for i in range(min(n_samples, len(dataset))):
            # Adjust field name based on your dataset structure
            #prompts.append(dataset[i]['prompt'])  # ← VERIFY FIELD NAME
            prompts.append(dataset[i]['input'].strip())
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return prompts[:n_samples]



# ============= VOCABULARY OVERLAP ANALYSIS =============

class VocabularyAnalyzer:
    """
    Analyzes vocabulary overlap between small (guidance) and large (base) models
    at nudging positions to understand why certain pairs fail.
    """
    
    def __init__(self, small_model_name, large_model_name, device='cuda'):
        """
        Args:
            small_model_name: HuggingFace model name for small model (e.g., 'Qwen/Qwen-1.8B-Chat')
            large_model_name: HuggingFace model name for large model (e.g., 'meta-llama/Llama-3-8B')
        """
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)
        self.large_tokenizer = AutoTokenizer.from_pretrained(large_model_name)
        
        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.large_model = AutoModelForCausalLM.from_pretrained(
            large_model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        self.device = device
        
    def get_nudging_positions(self, input_ids, large_logits, uncertainty_threshold=0.4):
        """
        Identify positions where NUDGING would intervene based on uncertainty.
        
        Args:
            input_ids: Token IDs of current sequence
            large_logits: Logits from large model
            uncertainty_threshold: Threshold for intervention (γ in NUDGING paper)
            
        Returns:
            List of positions where nudging would occur
        """
        probs = torch.softmax(large_logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        
        # Positions where top-1 probability < threshold = uncertain
        nudging_positions = (max_probs < uncertainty_threshold).nonzero(as_tuple=True)[0]
        return nudging_positions.cpu().tolist()
    

    def analyze_single_example(self, prompt, uncertainty_threshold=0.4, max_new_tokens=100):
        """
        Analyze vocabulary overlap for a single generation example.
        
        Returns:
            Dictionary with overlap metrics
        """
        # Start with prompt text
        current_text = prompt
        
        results = {
            'positions': [],
            'small_tokens': [],
            'small_token_strs': [],
            'ranks_in_large': [],
            'large_top5_tokens': [],
            'in_top_10': [],
            'in_top_50': [],
            'in_top_100': [],

            'in_top_p_90': [],

        }
        
        # Generate token by token
        for step in range(max_new_tokens):
            # Re-tokenize current text for both models
            # This ensures both models are always synchronized
            small_inputs = self.small_tokenizer(current_text, return_tensors='pt').to(self.device)
            large_inputs = self.large_tokenizer(current_text, return_tensors='pt').to(self.device)
            
            # Get logits from both models
            with torch.no_grad():
                small_outputs = self.small_model(**small_inputs)
                large_outputs = self.large_model(**large_inputs)
            
            small_logits = small_outputs.logits[0, -1, :]  # Last position
            large_logits = large_outputs.logits[0, -1, :]
            
            # Check if this is a nudging position
            small_probs = torch.softmax(small_logits, dim=-1)
            large_probs = torch.softmax(large_logits, dim=-1)
            
            large_max_prob = large_probs.max().item()
            
            if large_max_prob < uncertainty_threshold:  # Would nudge here
                # Small model's top choice
                small_token_id = small_logits.argmax().item()
                small_token_str = self.small_tokenizer.decode([small_token_id])
                
                # Convert small model's token to large model's vocabulary
                # Try to find equivalent token in large model
                try:
                    large_token_id = self.large_tokenizer.encode(
                        small_token_str, 
                        add_special_tokens=False
                    )[0]
                except:
                    # Token doesn't exist in large vocab
                    large_token_id = None
                    rank_in_large = float('inf')
                
                if large_token_id is not None:
                    # Find rank of this token in large model's distribution
                    large_sorted_indices = torch.argsort(large_probs, descending=True)
                    rank_in_large = (large_sorted_indices == large_token_id).nonzero(as_tuple=True)[0].item()

                    top_p_set = get_top_p_set(large_probs, p=0.90)   # ← add
                    in_p90 = large_token_id in top_p_set              # ← add

                else:
                    rank_in_large = float('inf')

                    in_p90 = False                                     # ← add
                
                results['in_top_p_90'].append(in_p90)

                #top_p_set = get_top_p_set(large_probs, p=0.90)
                #results['in_top_p_90'].append(large_token_id in top_p_set if large_token_id is not None else False)


                # Get large model's top 5 preferences
                large_top5_ids = large_logits.topk(5).indices.cpu().tolist()
                large_top5_strs = [self.large_tokenizer.decode([tid]) for tid in large_top5_ids]
                
                # Record results
                results['positions'].append(step)
                results['small_tokens'].append(small_token_id)
                results['small_token_strs'].append(small_token_str)
                results['ranks_in_large'].append(rank_in_large)
                results['large_top5_tokens'].append(large_top5_strs)
                results['in_top_10'].append(rank_in_large < 10)
                results['in_top_50'].append(rank_in_large < 50)
                results['in_top_100'].append(rank_in_large < 100)
            
            # Generate next token from LARGE model
            next_token_id = large_logits.argmax().item()
            next_token_str = self.large_tokenizer.decode([next_token_id])
            
            # Update text - both models will use this in next iteration
            current_text += next_token_str
            
            # Stop at EOS
            if next_token_id == self.large_tokenizer.eos_token_id:
                break
        
        return results


    def analyze_single_example_old(self, prompt, uncertainty_threshold=0.4, max_new_tokens=100):
        """
        Analyze vocabulary overlap for a single generation example.
        
        Returns:
            Dictionary with overlap metrics
        """
        # Tokenize prompt
        small_inputs = self.small_tokenizer(prompt, return_tensors='pt').to(self.device)
        large_inputs = self.large_tokenizer(prompt, return_tensors='pt').to(self.device)
        
        results = {
            'positions': [],
            'small_tokens': [],
            'small_token_strs': [],
            'ranks_in_large': [],
            'large_top5_tokens': [],
            'in_top_10': [],
            'in_top_50': [],
            'in_top_100': [],
        }
        
        # Generate token by token
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get logits from both models
                small_outputs = self.small_model(**small_inputs)
                large_outputs = self.large_model(**large_inputs)
                
                small_logits = small_outputs.logits[0, -1, :]  # Last position
                large_logits = large_outputs.logits[0, -1, :]
                
                # Check if this is a nudging position
                small_probs = torch.softmax(small_logits, dim=-1)
                large_probs = torch.softmax(large_logits, dim=-1)
                
                large_max_prob = large_probs.max().item()
                
                if large_max_prob < uncertainty_threshold:  # Would nudge here
                    # Small model's top choice
                    small_token_id = small_logits.argmax().item()
                    small_token_str = self.small_tokenizer.decode([small_token_id])
                    
                    # Convert small model's token to large model's vocabulary
                    # Try to find equivalent token in large model
                    try:
                        large_token_id = self.large_tokenizer.encode(
                            small_token_str, 
                            add_special_tokens=False
                        )[0]
                    except:
                        # Token doesn't exist in large vocab
                        large_token_id = None
                        rank_in_large = float('inf')
                    
                    if large_token_id is not None:
                        # Find rank of this token in large model's distribution
                        large_sorted_indices = torch.argsort(large_probs, descending=True)
                        rank_in_large = (large_sorted_indices == large_token_id).nonzero(as_tuple=True)[0].item()
                    else:
                        rank_in_large = float('inf')
                    
                    # Get large model's top 5 preferences
                    large_top5_ids = large_logits.topk(5).indices.cpu().tolist()
                    large_top5_strs = [self.large_tokenizer.decode([tid]) for tid in large_top5_ids]
                    
                    # Record results
                    results['positions'].append(step)
                    results['small_tokens'].append(small_token_id)
                    results['small_token_strs'].append(small_token_str)
                    results['ranks_in_large'].append(rank_in_large)
                    results['large_top5_tokens'].append(large_top5_strs)
                    results['in_top_10'].append(rank_in_large < 10)
                    results['in_top_50'].append(rank_in_large < 50)
                    results['in_top_100'].append(rank_in_large < 100)
                
                # Continue generation with large model's choice
                next_token = large_logits.argmax().unsqueeze(0).unsqueeze(0)
                large_inputs['input_ids'] = torch.cat([large_inputs['input_ids'], next_token], dim=1)
                
                # Also update small model input (try to sync)
                try:
                    small_next = self.small_tokenizer.encode(
                        self.large_tokenizer.decode([next_token.item()]),
                        add_special_tokens=False,
                        return_tensors='pt'
                    ).to(self.device)
                    small_inputs['input_ids'] = torch.cat([small_inputs['input_ids'], small_next], dim=1)
                except:
                    break  # Token incompatibility, stop
                
                # Stop if EOS
                if next_token.item() == self.large_tokenizer.eos_token_id:
                    break
        
        return results
    
    def compute_overlap_metrics(self, results):
        """
        Compute aggregate overlap metrics from analysis results.
        """
        if len(results['ranks_in_large']) == 0:
            return {
                'top_10_overlap': 0.0,
                'top_50_overlap': 0.0,
                'top_100_overlap': 0.0,
                'avg_rank': float('inf'),
                'median_rank': float('inf'),
                'num_nudging_positions': 0,

                'top_p_90_overlap': 0.0,   # ← add this
            }
        
        valid_ranks = [r for r in results['ranks_in_large'] if r != float('inf')]
        
        metrics = {
            'top_10_overlap': np.mean(results['in_top_10']) * 100,
            'top_50_overlap': np.mean(results['in_top_50']) * 100,
            'top_100_overlap': np.mean(results['in_top_100']) * 100,
            'avg_rank': np.mean(valid_ranks) if valid_ranks else float('inf'),
            'median_rank': np.median(valid_ranks) if valid_ranks else float('inf'),
            'num_nudging_positions': len(results['positions']),
            'vocab_incompatible_rate': (len(results['ranks_in_large']) - len(valid_ranks)) / len(results['ranks_in_large']) * 100,

            'top_p_90_overlap': np.mean(results['in_top_p_90']) * 100 if results['in_top_p_90'] else 0.0,

        }
        
        return metrics
    
    def analyze_dataset(self, prompts, uncertainty_threshold=0.4):
        """
        Analyze vocabulary overlap across multiple examples.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            Aggregate metrics and detailed results
        """
        all_results = []
        
        for prompt in tqdm(prompts, desc="Analyzing vocabulary overlap"):
            result = self.analyze_single_example(prompt, uncertainty_threshold)
            all_results.append(result)
        
        # Aggregate metrics
        aggregate = {
            'top_10_overlap': [],
            'top_50_overlap': [],
            'top_100_overlap': [],
            'avg_rank': [],
            'median_rank': [],
            'total_nudging_positions': 0,

            'top_p_90_overlap': []
            
        }
        
        for result in all_results:
            metrics = self.compute_overlap_metrics(result)
            aggregate['top_10_overlap'].append(metrics['top_10_overlap'])
            aggregate['top_50_overlap'].append(metrics['top_50_overlap'])
            aggregate['top_100_overlap'].append(metrics['top_100_overlap'])

            aggregate['top_p_90_overlap'].append(metrics['top_p_90_overlap'])

            if metrics['avg_rank'] != float('inf'):
                aggregate['avg_rank'].append(metrics['avg_rank'])
            if metrics['median_rank'] != float('inf'):
                aggregate['median_rank'].append(metrics['median_rank'])
            aggregate['total_nudging_positions'] += metrics['num_nudging_positions']
        
        summary = {
            'mean_top_10_overlap': np.mean(aggregate['top_10_overlap']),
            'mean_top_50_overlap': np.mean(aggregate['top_50_overlap']),
            'mean_top_100_overlap': np.mean(aggregate['top_100_overlap']),
            'mean_avg_rank': np.mean(aggregate['avg_rank']) if aggregate['avg_rank'] else float('inf'),
            'mean_median_rank': np.mean(aggregate['median_rank']) if aggregate['median_rank'] else float('inf'),
            'total_nudging_positions': aggregate['total_nudging_positions'],

            'mean_top_p_90_overlap': np.mean(aggregate['top_p_90_overlap']),

            
        }
        
        return summary, all_results


# ============= USAGE EXAMPLE =============

def compare_model_pairs():
    """
    Compare vocabulary overlap across different model pairs.
    """
    N_SAMPLES_PER_DATASET = 30  # ← YOUR CHOICE: 30, 50, or 100

    # Load prompts from multiple benchmarks
    print("Loading benchmark data...")
    gsm8k_prompts = load_benchmark_prompts('gsm8k', n_samples=N_SAMPLES_PER_DATASET)
    mmlu_prompts = load_benchmark_prompts('mmlu', n_samples=N_SAMPLES_PER_DATASET)
    truthfulqa_prompts = load_benchmark_prompts('truthfulqa', n_samples=N_SAMPLES_PER_DATASET)
    arc_prompts = load_benchmark_prompts('arc', n_samples=N_SAMPLES_PER_DATASET)
    xstest_prompts = load_benchmark_prompts('xstest', n_samples=N_SAMPLES_PER_DATASET)
    justeval_prompts = load_benchmark_prompts('justeval', n_samples=N_SAMPLES_PER_DATASET)
    
    # Combine all (total: 6 × N_SAMPLES)
    test_prompts = (gsm8k_prompts + mmlu_prompts + truthfulqa_prompts + 
                   arc_prompts + xstest_prompts + justeval_prompts)
    
    print(f"Loaded {len(test_prompts)} test prompts ({N_SAMPLES_PER_DATASET} per dataset)")
    # Combine for diverse testing (total 80 samples)
    #test_prompts = gsm8k_prompts + mmlu_prompts + arc_prompts
    
    #print(f"Loaded {len(test_prompts)} test prompts")
    
    model_pairs = [
        #('Qwen/Qwen2.5-1.5B-Instruct', 'meta-llama/Meta-Llama-3-8B', 'Qwen→Llama'),
        #('meta-llama/Llama-3.2-1B-Instruct', 'google/gemma-2-9b', 'Llama→Gemma'),
        #('google/gemma-2-2b-it', 'meta-llama/Meta-Llama-3-8B', 'Gemma→Llama'),
        #('meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Meta-Llama-3-8B', 'Llama→Llama (within)'),

        # Within-family (select representative)
        ('meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.1-8B', 'Llama-3 1B→8B'),
        ('google/gemma-3-1b-it', 'google/gemma-2-9b', 'Gemma-3 1B→9B'),
        ('Qwen/Qwen3-1.7B', 'Qwen/Qwen3-8B-Base', 'Qwen-3 1.7B→8B'),
        
        # Cross-family (key comparisons)
        ('meta-llama/Llama-3.2-1B-Instruct', 'google/gemma-2-9b', 'Llama→Gemma'),
        ('google/gemma-3-1b-it', 'meta-llama/Llama-3.1-8B', 'Gemma→Llama'),
        ('Qwen/Qwen3-1.7B', 'meta-llama/Llama-3.1-8B', 'Qwen→Llama'),
        ('meta-llama/Llama-3.2-1B-Instruct', 'Qwen/Qwen3-8B-Base', 'Llama→Qwen'),
        ('google/gemma-3-1b-it', 'Qwen/Qwen3-8B-Base', 'Gemma→Qwen'),  # ADD THIS
        ('Qwen/Qwen3-1.7B', 'google/gemma-2-9b', 'Qwen→Gemma'),  # ADD THIS

    ]
    
    results = {}
    
    for small_name, large_name, pair_label in model_pairs:
        print(f"\n{'='*60}")
        print(f"Analyzing: {pair_label}")
        print(f"{'='*60}")
        
        analyzer = VocabularyAnalyzer(small_name, large_name, device='cuda')
        summary, detailed = analyzer.analyze_dataset(test_prompts)
        
        results[pair_label] = summary
        
        print(f"\nResults for {pair_label}:")
        print(f"  Top-10 Overlap:  {summary['mean_top_10_overlap']:.1f}%")
        print(f"  Top-50 Overlap:  {summary['mean_top_50_overlap']:.1f}%")
        print(f"  Top-100 Overlap: {summary['mean_top_100_overlap']:.1f}%")
        print(f"  Avg Rank:        {summary['mean_avg_rank']:.1f}")
        print(f"  Median Rank:     {summary['mean_median_rank']:.1f}")
        print(f"  Nudging Positions: {summary['total_nudging_positions']}")
        
        # Clean up memory
        del analyzer
        torch.cuda.empty_cache()
    
    return results



def visualize_overlap_comparison(results):
    """
    Create visualizations comparing vocabulary overlap across pairs.
    """
    
    # Performance data for all benchmarks
    performance_data = {
        'gsm8k': {
            'Llama-3 1B→8B': 0.58, 'Gemma-3 1B→9B': 0.66, 'Qwen-3 1.7B→8B': 0.12,
            'Llama→Gemma': 0.67, 'Gemma→Llama': 0.59, 'Llama→Qwen': 0.87,
            'Gemma→Qwen': 0.82, 'Qwen→Llama': 0.27, 'Qwen→Gemma': 0.54,
        },
        'mmlu': {
            'Llama-3 1B→8B': 0.49, 'Gemma-3 1B→9B': 0.49, 'Qwen-3 1.7B→8B': 0.21,
            'Llama→Gemma': 0.60, 'Gemma→Llama': 0.47, 'Llama→Qwen': 0.63,
            'Gemma→Qwen': 0.70, 'Qwen→Llama': 0.41, 'Qwen→Gemma': 0.22,
        },
        'truthfulqa': {
            'Llama-3 1B→8B': 0.51, 'Gemma-3 1B→9B': 0.43, 'Qwen-3 1.7B→8B': 0.52,
            'Llama→Gemma': 0.47, 'Gemma→Llama': 0.45, 'Llama→Qwen': 0.61,
            'Gemma→Qwen': 0.59, 'Qwen→Llama': 0.48, 'Qwen→Gemma': 0.40,
        },
        'arc': {
            'Llama-3 1B→8B': 0.71, 'Gemma-3 1B→9B': 0.72, 'Qwen-3 1.7B→8B': 0.18,
            'Llama→Gemma': 0.75, 'Gemma→Llama': 0.64, 'Llama→Qwen': 0.84,
            'Gemma→Qwen': 0.86, 'Qwen→Llama': 0.46, 'Qwen→Gemma': 0.47,
        },
        'xstest': {
            'Llama-3 1B→8B': 0.12, 'Gemma-3 1B→9B': 0.10, 'Qwen-3 1.7B→8B': 0.08,
            'Llama→Gemma': 0.10, 'Gemma→Llama': 0.08, 'Llama→Qwen': 0.18,
            'Gemma→Qwen': 0.09, 'Qwen→Llama': 0.03, 'Qwen→Gemma': 0.02,
        },
        'justeval': {
            'Llama-3 1B→8B': 4.86, 'Gemma-3 1B→9B': 4.92, 'Qwen-3 1.7B→8B': 4.20,
            'Llama→Gemma': 4.82, 'Gemma→Llama': 4.89, 'Llama→Qwen': 4.93,
            'Gemma→Qwen': 4.98, 'Qwen→Llama': 3.22, 'Qwen→Gemma': 3.19,
        },
    }
    
    #fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 grid
#    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 8 plots total
#    axes = axes.flatten()
    # Abbreviate pair names
#    abbreviated_results = {}
#    for pair, metrics in results.items():
#        abbrev_pair = pair.replace('Llama', 'L').replace('Gemma', 'G').replace('Qwen', 'Q')
#        abbreviated_results[abbrev_pair] = metrics
#    results = abbreviated_results
    
#    fig, axes = plt.subplots(1, 8, figsize=(32, 4))  # ← One row, 8 columns
    fig, axes = plt.subplots(2, 4, figsize=(30, 14))  # ← ADD THIS
    axes = axes.flatten()  # ← ADD THIS (if you removed it)
    # Remove: axes = axes.flatten()
    
    pairs = list(results.keys())
    
    # Plot 1: Top-K Overlap Rates
    ax = axes[0]
    x = np.arange(len(pairs))
    width = 0.25
    
    top10 = [results[p]['mean_top_10_overlap'] for p in pairs]
    top50 = [results[p]['mean_top_50_overlap'] for p in pairs]
    top100 = [results[p]['mean_top_100_overlap'] for p in pairs]
    

    #top_p90 = [results[p]['mean_top_p_90_overlap'] for p in pairs]
    #ax.bar(x + 1.5*width, top_p90, width, label='Top-p(0.9)', color='#9b59b6')


    ax.bar(x - width, top10, width, label='Top-10', color='#e74c3c')
    ax.bar(x, top50, width, label='Top-50', color='#f39c12')
    ax.bar(x + width, top100, width, label='Top-100', color='#2ecc71')
    
    ax.set_xlabel('Model Pair', fontsize=26, fontweight='bold')#0)
    ax.set_ylabel('Overlap Rate (%)', fontsize=28, fontweight='bold')#0)
    ax.set_title('Vocabulary Overlap at Top K', fontsize=28, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=26)#8)
    ax.legend(fontsize=26)#8)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', labelsize=24)
    
    # Plot 2: Average Rank
    ax = axes[1]
    avg_ranks = [results[p]['mean_avg_rank'] for p in pairs]
    colors = ['#e74c3c' if 'Qwen→' in p else '#2ecc71' for p in pairs]
    
    ax.bar(pairs, avg_ranks, color=colors, alpha=0.7)
    ax.set_xlabel('Model Pair', fontsize=26, fontweight='bold')#0) #22
    ax.set_ylabel('Average Rank in Base', fontsize=28, fontweight='bold')#0)
    ax.set_title('Ranking of Suggested Tokens', fontsize=28, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=26)#8)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', labelsize=24)
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, label='Rank 100 threshold')
    ax.legend(fontsize=26)#8)
    
    # Plots 3-8: Correlation for each benchmark
    benchmarks = ['gsm8k', 'mmlu', 'truthfulqa', 'arc', 'xstest', 'justeval']
    benchmark_names = ['GSM8K', 'MMLU', 'TruthfulQA', 'ARC-Challenge', 'XSTest', 'JustEval-Safe']
    
    from scipy.stats import pearsonr
    
    for idx, (benchmark, benchmark_name) in enumerate(zip(benchmarks, benchmark_names)):
        ax = axes[idx + 2]  # Plots 3-8
        
        overlap_rates = [results[p]['mean_top_50_overlap'] for p in pairs]
        performance = [performance_data[benchmark].get(p, 0) for p in pairs]
        
        # Color by family
        colors_scatter = []
        for p in pairs:
            if 'Qwen→' in p:
                colors_scatter.append('#e74c3c')  # Red for Qwen providing guidance
            elif '→' in p:
                colors_scatter.append('#f39c12')  # Orange for cross-family
            else:
                colors_scatter.append('#2ecc71')  # Green for within-family
        
        ax.scatter(overlap_rates, performance, s=80, alpha=0.7, c=colors_scatter)
        
        # Add labels for key points (optional - comment out if too cluttered)
        # for i, pair in enumerate(pairs):
        #     ax.annotate(pair, (overlap_rates[i], performance[i]), 
        #                fontsize=6, ha='right', alpha=0.6)
        
        # Correlation
        valid_pairs = [(o, p) for o, p in zip(overlap_rates, performance) if p > 0]
        if len(valid_pairs) > 2:
            valid_overlap = [x[0] for x in valid_pairs]
            valid_perf = [x[1] for x in valid_pairs]
            r, p_val = pearsonr(valid_overlap, valid_perf)
            
            # Add regression line if correlation is significant
            if p_val < 0.1:  # Show line if p < 0.1
                z = np.polyfit(valid_overlap, valid_perf, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(min(valid_overlap), max(valid_overlap), 100)
                ax.plot(x_line, p_fit(x_line), 'k--', alpha=0.3, linewidth=1)
            
            # Display correlation
            sig_marker = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')
            ax.text(0.05, 0.95, f'r={r:.2f}{sig_marker}\np={p_val:.3f}', 
                   transform=ax.transAxes, va='top', fontsize=26, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)) #20
        
        ax.set_xlabel('Top-50 Vocab Overlap (%)', fontsize=26, fontweight='bold')#9)
        #ax.set_ylabel(f'{benchmark_name} Performance', fontsize=26, fontweight='bold')#9)
        ax.set_ylabel(f'Performance', fontsize=26, fontweight='bold')#9)
        ax.set_title(f'Overlap on {benchmark_name}', fontsize=26, fontweight='bold')
        ax.tick_params(labelsize=24)#16)  # ← Add this line for tick labels
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
#    plt.savefig('vocabulary_overlap_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('vocabulary_overlap_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print correlation summary
    print("\n" + "="*80)
    print("CORRELATION SUMMARY: Vocabulary Overlap vs Performance")
    print("="*80)
    print(f"{'Benchmark':<20} {'Correlation (r)':<20} {'P-value':<15} {'Significance':<15}")
    print("-"*80)
    
    for benchmark, benchmark_name in zip(benchmarks, benchmark_names):
        overlap_rates = [results[p]['mean_top_50_overlap'] for p in pairs]
        performance = [performance_data[benchmark].get(p, 0) for p in pairs]
        
        valid_pairs = [(o, p) for o, p in zip(overlap_rates, performance) if p > 0]
        if len(valid_pairs) > 2:
            valid_overlap = [x[0] for x in valid_pairs]
            valid_perf = [x[1] for x in valid_pairs]
            r, p_val = pearsonr(valid_overlap, valid_perf)
            
            if p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            elif p_val < 0.1:
                sig = '†'
            else:
                sig = 'ns'
            
            print(f"{benchmark_name:<20} {r:>8.3f} {'':<11} {p_val:>8.4f} {'':<6} {sig:<15}")
    
    print("-"*80)
    print("**: p<0.01 (highly significant), *: p<0.05 (significant), †: p<0.1 (marginally sig), ns: not significant")


## What This Does:

#**Layout:**
#```
#[Plot 1: Overlap Bars]  [Plot 2: Avg Rank]
#[Plot 3: GSM8K Corr]    [Plot 4: MMLU Corr]
#[Plot 5: TruthQA Corr]  [Plot 6: ARC Corr]
#[Plot 7: XSTest Corr]   [Plot 8: JustEval Corr]




def visualize_overlap_comparison_old(results):
    """
    Create visualizations comparing vocabulary overlap across pairs.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pairs = list(results.keys())
    
    # Plot 1: Top-K Overlap Rates
    ax = axes[0]
    x = np.arange(len(pairs))
    width = 0.25
    
    top10 = [results[p]['mean_top_10_overlap'] for p in pairs]
    top50 = [results[p]['mean_top_50_overlap'] for p in pairs]
    top100 = [results[p]['mean_top_100_overlap'] for p in pairs]
    
    ax.bar(x - width, top10, width, label='Top-10', color='#e74c3c')
    ax.bar(x, top50, width, label='Top-50', color='#f39c12')
    ax.bar(x + width, top100, width, label='Top-100', color='#2ecc71')
    
    ax.set_xlabel('Model Pair')
    ax.set_ylabel('Overlap Rate (%)')
    ax.set_title('Vocabulary Overlap at Different Rank Thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Average Rank
    ax = axes[1]
    avg_ranks = [results[p]['mean_avg_rank'] for p in pairs]
    colors = ['#e74c3c' if 'Qwen' in p and '→' in p else '#2ecc71' for p in pairs]
    
    ax.bar(pairs, avg_ranks, color=colors, alpha=0.7)
    ax.set_xlabel('Model Pair')
    ax.set_ylabel('Average Rank in Large Model')
    ax.set_title('How High-Ranked Are Nudged Tokens?')
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=100, color='red', linestyle='--', label='Rank 100 threshold')
    ax.legend()
    
    # Plot 3: Correlation with Performance
    # You'll need to add your actual GSM8K performance here
    ax = axes[2]
    
    # Example - replace with your actual data
    gsm8k_performance = {
        'Qwen→Llama': 0.27,
        'Llama→Gemma': 0.67,
        'Gemma→Llama': 0.59,
        'Llama->Qwen': 0.87, 
        'Llama-3 1B→8B': 0.58,
        'Gemma-3 1B→9B': 0.66,
        'Qwen-3 1.7B→8B': 0.12,
    }

    
    




    overlap_rates = [results[p]['mean_top_50_overlap'] for p in pairs]
    performance = [gsm8k_performance.get(p, 0) for p in pairs]
    
    ax.scatter(overlap_rates, performance, s=100, alpha=0.7)
    for i, pair in enumerate(pairs):
        ax.annotate(pair, (overlap_rates[i], performance[i]), 
                   fontsize=8, ha='right')
    
    # Correlation
    from scipy.stats import pearsonr
    if len(overlap_rates) > 2:
        r, p = pearsonr(overlap_rates, performance)
        ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.3f}', 
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Top-50 Vocabulary Overlap (%)')
    ax.set_ylabel('GSM8K Performance')
    ax.set_title('Overlap vs Performance')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig('vocabulary_overlap_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('vocabulary_overlap_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


# ============= RUN ANALYSIS =============

#if __name__ == "__main__":
    # Run comparison
#    results = compare_model_pairs()
    
    # Visualize
#    visualize_overlap_comparison(results)
    
    # Print summary table
#    print("\n" + "="*80)
#    print("VOCABULARY OVERLAP SUMMARY")
#    print("="*80)
#    print(f"{'Model Pair':<25} {'Top-10%':<10} {'Top-50%':<10} {'Top-100%':<10} {'Avg Rank':<10}")
#    print("-"*80)
#    for pair, metrics in results.items():
#        print(f"{pair:<25} "
#              f"{metrics['mean_top_10_overlap']:>8.1f}% "
#              f"{metrics['mean_top_50_overlap']:>8.1f}% "
#              f"{metrics['mean_top_100_overlap']:>8.1f}% "
#              f"{metrics['mean_avg_rank']:>8.1f}")



# ============= RUN ANALYSIS =============

if __name__ == "__main__":
    RESULTS_FILE = 'vocab_analysis_results.json'
    
    # Try to load existing results
    results = load_results(RESULTS_FILE)
    
    if results is None:
        print("No saved results found. Running analysis...")
        # Run comparison (takes hours)
        results = compare_model_pairs()
        
        # Save immediately after completion
        save_results(results, RESULTS_FILE)
    else:
        print("Using saved results. Delete the file to re-run analysis.")
        user_input = input("Do you want to re-run analysis? (yes/no): ").strip().lower()
        if user_input == 'yes':
            print("Re-running analysis...")
            results = compare_model_pairs()
            save_results(results, RESULTS_FILE)
    
    # SAVE AGAIN before graphing (insurance)
    #save_results(results, RESULTS_FILE)  # ← ADD THIS LINE

    # Visualize (can be re-run multiple times without re-analyzing)
    visualize_overlap_comparison(results)
    

    # Reviewer response: top-p analysis
    generate_reviewer_topp_response(results)


    # Print summary table
    print("\n" + "="*80)
    print("VOCABULARY OVERLAP SUMMARY")
    print("="*80)
    print(f"{'Model Pair':<25} {'Top-10%':<10} {'Top-50%':<10} {'Top-100%':<10} {'Avg Rank':<10}")
    print("-"*80)
    for pair, metrics in results.items():
        print(f"{pair:<25} "
              f"{metrics['mean_top_10_overlap']:>8.1f}% "
              f"{metrics['mean_top_50_overlap']:>8.1f}% "
              f"{metrics['mean_top_100_overlap']:>8.1f}% "
              f"{metrics['mean_avg_rank']:>8.1f}")