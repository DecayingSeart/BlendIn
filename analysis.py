import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


# Set Times font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # For math symbols
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# ========== ADD THESE LINES ==========
# Ensure fonts are embedded for publication
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
plt.rcParams['ps.fonttype'] = 42   # For EPS compatibility
# =====================================

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ============= DATA PREPARATION =============

cross_family_pairs = [
    'Llama-3 1B→Gemma-3 9B',
    'Llama-3 1B→Qwen-3 8B',
    'Gemma-3 1B→Llama-3 8B',
    'Gemma-3 1B→Qwen-3 8B',
    'Qwen-3 1.7B→Llama-3 8B',
    'Qwen-3 1.7B→Gemma-3 9B'
]

cross_family_labels = ['Cross']*6  # or keep specific like ['Llama→Gemma', ...]

cross_gap_ratios = [9, 8, 8, 8, 5, 5]

# GSM8K: num / % / metric
cross_gsm8k_num = [11.0, 3.13, 18.09, 6.71, 75.77, 77.83]
cross_gsm8k_pct = [9.2, 2.0, 14.8, 3.8, 22.2, 23.1]
cross_gsm8k = [0.67, 0.87, 0.59, 0.82, 0.27, 0.54]

# MMLU
cross_mmlu_num = [24.76, 14.33, 50.61, 21.88, 99.04, 71.51]
cross_mmlu_pct = [13.8, 7.1, 18.5, 7.4, 28.7, 20.1]
cross_mmlu = [0.60, 0.63, 0.47, 0.70, 0.41, 0.22]

# TruthfulQA
cross_truthfulqa_num = [16.77, 8.03, 31.23, 15.79, 108.19, 111.7]
cross_truthfulqa_pct = [17.2, 11.1, 19.2, 6.7, 31.9, 31.5]
cross_truthfulqa = [0.47, 0.61, 0.45, 0.59, 0.48, 0.40]

# ARC-Challenge
cross_arc_num = [22.08, 12.85, 40.75, 18.3, 104.52, 94.17]
cross_arc_pct = [12.6, 5.9, 17.4, 7.6, 29.5, 24.9]
cross_arc = [0.75, 0.84, 0.64, 0.86, 0.46, 0.47]

# XSTest
cross_xstest_num = [30.04, 18.65, 93.91, 34.75, 112.34, 114.05]
cross_xstest_pct = [20.3, 10.9, 27.5, 14.0, 33.1, 34.3]
cross_xstest = [0.10, 0.18, 0.08, 0.09, 0.03, 0.02]

# JustEval-Safe
cross_justeval_num = [8.87, 5.2, 100.99, 30.23, 128.01, 122.28]
cross_justeval_pct = [24.9, 8.1, 27.8, 10.9, 35.9, 37.7]
cross_justeval = [4.82, 4.93, 4.89, 4.98, 3.22, 3.19]


# Table 1: NUDGING results (nudging_num / nudging_% / metric)
nudging_data = {
    'model_pair': [
        'Llama-3 1B→3B', 'Llama-3 1B→8B', 'Llama-3 3B→8B',
        'Gemma-3 1B→4B', 'Gemma-3 1B→9B', 'Gemma-3 4B→9B',
        'Qwen-3 1.7B→4B', 'Qwen-3 1.7B→8B', 'Qwen-3 4B→8B'
    ] + cross_family_pairs,
    'family': ['Llama-3']*3 + ['Gemma-3']*3 + ['Qwen-3']*3 + ['Cross']*6,
    'gap_ratio': [3, 8, 3, 4, 9, 2.5, 2.5, 5, 2] + cross_gap_ratios,
    'is_cross_family': [False]*9 + [True]*6,  # Add True for cross-family pairs
    
    # GSM8K: num / % / metric
    'gsm8k_num': [21.99, 17.43, 16.56, 16.54, 14.78, 13.28, 55.5, 46.88, 50.77] + cross_gsm8k_num,
    'gsm8k_pct': [16.2, 14.4, 12.2, 10.6, 9.9, 10.0, 16.2, 13.1, 14.3] + cross_gsm8k_pct,
    'gsm8k': [0.31, 0.58, 0.61, 0.54, 0.66, 0.76, 0.20, 0.12, 0.18] + cross_gsm8k,
    
    # MMLU
    'mmlu_num': [50.53, 36.25, 33.3, 54.27, 46.58, 40.26, 50.55, 40.26, 48.59] + cross_mmlu_num,
    'mmlu_pct': [21.2, 18.1, 16.2, 20.9, 19.2, 16.2, 14.3, 11.2, 13.0] + cross_mmlu_pct,
    'mmlu': [0.45, 0.49, 0.57, 0.47, 0.49, 0.61, 0.31, 0.21, 0.49] + cross_mmlu,
    
    # TruthfulQA
    'truthfulqa_num': [19.03, 16.07, 16.25, 34.31, 26.32, 29.24, 72.24, 56.47, 60.5] + cross_truthfulqa_num ,
    'truthfulqa_pct': [19.7, 20.0, 21.7, 18.9, 17.6, 20.9, 20.0, 15.1, 16.7]+cross_truthfulqa_pct ,
    'truthfulqa': [0.48, 0.51, 0.59, 0.43, 0.43, 0.53, 0.53, 0.52, 0.56] + cross_truthfulqa,
    
    # ARC-Challenge
    'arc_num': [36.28, 26.75, 29.73, 44.77, 36.73, 32.61, 44.59, 32.56, 48.94] + cross_arc_num,
    'arc_pct': [18.1, 16.3, 16.5, 20.8, 17.8, 15.2, 12.9, 9.5, 12.7] + cross_arc_pct,
    'arc': [0.58, 0.71, 0.77, 0.61, 0.72, 0.79, 0.27, 0.18, 0.80]+cross_arc ,
    
    # XSTest
    'xstest_num': [38.29, 29.17, 29.44, 101.72, 95.29, 93.7, 97.67, 98.16, 95.66] + cross_xstest_num,
    'xstest_pct': [24.8, 22.2, 22.6, 30.3, 29.3, 28.0, 27.8, 26.7, 25.2] + cross_xstest_pct,
    'xstest': [0.10, 0.12, 0.15, 0.07, 0.10, 0.06, 0.06, 0.08, 0.14] + cross_xstest,
    
    # JustEval-Safe
    'justeval_num': [12.56, 9.97, 8.74, 107.85, 101.58, 105.87, 110.68, 104.15, 103.73]+cross_justeval_num,
    'justeval_pct': [31.8, 35.0, 31.8, 30.0, 29.7, 29.1, 31.0, 27.8, 27.0] + cross_justeval_pct,
    'justeval': [4.82, 4.86, 4.93, 4.85, 4.92, 4.96, 3.88, 4.2, 4.78] + cross_justeval,
}

# Table 2: Base model performance
base_data = {
    'Llama-3-3B-pt': {'gsm8k': 0.04, 'mmlu': 0.53, 'truthfulqa': 0.56, 'arc': 0.65, 'xstest': 0, 'justeval': 3.74},
    'Llama-3-3B-it': {'gsm8k': 0.75, 'mmlu': 0.56, 'truthfulqa': 0.65, 'arc': 0.82, 'xstest': 0.16, 'justeval': 4.96},
    'Llama-3-8B-pt': {'gsm8k': 0.11, 'mmlu': 0.59, 'truthfulqa': 0.58, 'arc': 0.79, 'xstest': 0, 'justeval': 3.12},
    'Llama-3-8B-it': {'gsm8k': 0.87, 'mmlu': 0.74, 'truthfulqa': 0.67, 'arc': 0.87, 'xstest': 0.15, 'justeval': 4.93},
    'Gemma-3-4B-pt': {'gsm8k': 0.08, 'mmlu': 0.15, 'truthfulqa': 0.39, 'arc': 0.19, 'xstest': 0.06, 'justeval': 3.15},
    'Gemma-3-4B-it': {'gsm8k': 0.82, 'mmlu': 0.59, 'truthfulqa': 0.67, 'arc': 0.85, 'xstest': 0.09, 'justeval': 4.98},
    'Gemma-3-9B-pt': {'gsm8k': 0.32, 'mmlu': 0.28, 'truthfulqa': 0.40, 'arc': 0.49, 'xstest': 0.01, 'justeval': 3.0},
    'Gemma-3-9B-it': {'gsm8k': 0.86, 'mmlu': 0.72, 'truthfulqa': 0.70, 'arc': 0.85, 'xstest': 0.14, 'justeval': 5.0},
    'Qwen-3-4B-it': {'gsm8k': 0.45, 'mmlu': 0.56, 'truthfulqa': 0.67, 'arc': 0.79, 'xstest': 0.08, 'justeval': 4.83},
    'Qwen-3-4B-pt': {'gsm8k': 0.66, 'mmlu': 0.43, 'truthfulqa': 0.64, 'arc': 0.64, 'xstest': 0.05, 'justeval': 4.76},
    'Qwen-3-8B-pt': {'gsm8k': 0.59, 'mmlu': 0.57, 'truthfulqa': 0.67, 'arc': 0.91, 'xstest': 0.10, 'justeval': 4.8},
    'Qwen-3-8B-it': {'gsm8k': 0.29, 'mmlu': 0.32, 'truthfulqa': 0.61, 'arc': 0.70, 'xstest': 0.08, 'justeval': 4.52},
}

df_nudging = pd.DataFrame(nudging_data)

# Calculate capability gaps (need MMLU scores for small/large models)
# Gap = Large_base_MMLU - Small_instruct_MMLU
capability_gaps = {
    'Llama-3 1B→3B': 0.53 - 0.47,  # Example: 3B MMLU - 1B MMLU 
    'Llama-3 1B→8B': 0.59 - 0.47,
    'Llama-3 3B→8B': 0.59 - 0.56,
    'Gemma-3 1B→4B': 0.15 - 0.42,  
    'Gemma-3 1B→9B': 0.28 - 0.42,
    'Gemma-3 4B→9B': 0.28 - 0.59,
    'Qwen-3 1.7B→4B': 0.43 - 0.33,  
    'Qwen-3 1.7B→8B': 0.57 - 0.33,
    'Qwen-3 4B→8B': 0.57 - 0.56,

    # NEW: Cross-family (you need to fill in 1B-it and 1.7B-it MMLU scores)
    'Llama-3 1B→Gemma-3 9B': 0.28 - 0.47,
    'Llama-3 1B→Qwen-3 8B': 0.57 - 0.47,
    'Gemma-3 1B→Llama-3 8B': 0.59 - 0.42,
    'Gemma-3 1B→Qwen-3 8B': 0.57 - 0.42,
    'Qwen-3 1.7B→Llama-3 8B': 0.59 - 0.33,
    'Qwen-3 1.7B→Gemma-3 9B': 0.28 - 0.33,
}

df_nudging['capability_gap'] = df_nudging['model_pair'].map(capability_gaps)

# ============= VISUALIZATION 1: Performance vs Capability Gap =============

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
tasks = ['gsm8k', 'mmlu', 'truthfulqa', 'arc', 'xstest', 'justeval']
task_names = ['GSM8K', 'MMLU', 'TruthfulQA', 'ARC-Challenge', 'XSTest', 'JustEval-Safe']
task_names_abb = ['G', 'M', 'T', 'A', 'X', 'J']


families_to_plot = ['Llama-3', 'Gemma-3', 'Qwen-3', 'Cross']
colors = {'Llama-3': 'blue', 'Gemma-3': 'orange', 'Qwen-3': 'green', 'Cross': 'red'}


for idx, (task, task_name) in enumerate(zip(tasks, task_names)):
    ax = axes[idx // 3, idx % 3]
    
    # Plot by family
    for family in families_to_plot:#['Llama-3', 'Gemma-3', 'Qwen-3']:
        family_data = df_nudging[df_nudging['family'] == family]
        #ax.scatter(family_data['capability_gap'], family_data[task], 
        #          label=family, s=100, alpha=0.7)
        if len(family_data) > 0:
            ax.scatter(family_data['capability_gap'], family_data[task], 
                label=family, color=colors[family], s=100, alpha=0.7)
        
        # Fit trend line
        if len(family_data) > 1:
            z = np.polyfit(family_data['capability_gap'], family_data[task], 1)
            p = np.poly1d(z)
            x_line = np.linspace(family_data['capability_gap'].min(), 
                                family_data['capability_gap'].max(), 100)
            ax.plot(x_line, p(x_line), '--', alpha=0.5)
    
    ax.set_xlabel('Capability Gap (MMLU Δ)', fontsize=10)
    ax.set_ylabel('Performance', fontsize=10)
    ax.set_title(task_name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_vs_capability_gap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============= VISUALIZATION 2: Nudging Intervention Rate vs Performance =============

#fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig, axes = plt.subplots(1, 6, figsize=(28, 4)) #24,10  # ← One row, 6 columns
#fig, axes = plt.subplots(2, 3, figsize=(22, 13))  # ← ADD THIS
axes = axes.flatten()  # ← ADD THIS

for idx, (task, task_name) in enumerate(zip(tasks, task_names)):
#    ax = axes[idx // 3, idx % 3]
    ax = axes[idx]  # ← Direct indexing (no //3, %3)
    
    # Scatter plot: nudging% vs performance
    scatter = ax.scatter(df_nudging[f'{task}_pct'], df_nudging[task],
                        c=df_nudging['capability_gap'], s=100, 
                        cmap='viridis', alpha=0.7)
    
    # Add correlation
    r, p = pearsonr(df_nudging[f'{task}_pct'], df_nudging[task])
    ax.text(0.05, 0.95, f'r={r:.2f}\np={p:.3f}', 
            transform=ax.transAxes, va='top', fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)) #16
    
#    ax.set_xlabel('Nudging % (intervention rate)', fontsize=10)
#    ax.set_ylabel('Performance', fontsize=10)
#    ax.set_title(task_name, fontsize=12, fontweight='bold')

#    ax.set_xlabel('Intervention Rate (%)', fontsize=28,fontweight='bold') #16 # ← Changed name + bigger
#    ax.set_ylabel('Performance', fontsize=24,fontweight='bold') #16 # ← Changed name + bigger
    ax.set_title(task_name, fontsize=20, fontweight='bold') #16 # ← Bigger
    ax.tick_params(labelsize=18)  # ← Bigger tick labels
    
    # Colorbar
#    if idx == 2:  # Add colorbar to one plot
#        plt.colorbar(scatter, ax=ax, label='Capability Gap')
    #if idx == 5:  # ← Rightmost plot (was idx==2)
    #    cbar = plt.colorbar(scatter, ax=ax, label='Capability Gap')
    #    cbar.ax.tick_params(labelsize=12)
    #    cbar.set_label('Capability Gap', fontsize=14)

# ========== ADD SHARED LABELS ==========
fig.supxlabel('Intervention Rate (%)', fontsize=22, fontweight='bold')  # ← Shared x-label
fig.supylabel('Performance', fontsize=22, fontweight='bold',x=0.03)  # ← Shared y-label

#plt.tight_layout()
plt.tight_layout(rect=[0.03, 0.05, 1, 1])  # ← [left, bottom, right, top]
# Leaves 3% margin on left, 5% on bottom

#plt.savefig('nudging_rate_vs_performance.png', dpi=300, bbox_inches='tight')
# Add colorbar spanning both rows on the right
#fig.colorbar(scatter, ax=axes.ravel().tolist(), 
#             label='Capability Gap', 
#             pad=0.02, aspect=30)
cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), pad=0.02, aspect=30)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Capability Gap', fontsize=18, fontweight='bold')  # ← Add this line
plt.savefig('nudging_rate_vs_performance.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# ============= VISUALIZATION 3: Performance Gain Heatmap =============

# Calculate performance gains (nudged - base)
# You'll need to map each model pair to its base performance

pair_to_base = {
    'Llama-3 1B→3B': 'Llama-3-3B-pt',      # 1B guides 3B base
    'Llama-3 1B→8B': 'Llama-3-8B-pt',      # 1B guides 8B base
    'Llama-3 3B→8B': 'Llama-3-8B-pt',      # 3B guides 8B base
    'Gemma-3 1B→4B': 'Gemma-3-4B-pt',
    'Gemma-3 1B→9B': 'Gemma-3-9B-pt',
    'Gemma-3 4B→9B': 'Gemma-3-9B-pt',
    'Qwen-3 1.7B→4B': 'Qwen-3-4B-pt',
    'Qwen-3 1.7B→8B': 'Qwen-3-8B-pt',
    'Qwen-3 4B→8B': 'Qwen-3-8B-pt',

    # NEW: Cross-family
    'Llama-3 1B→Gemma-3 9B': 'Gemma-3-9B-pt',
    'Llama-3 1B→Qwen-3 8B': 'Qwen-3-8B-pt',
    'Gemma-3 1B→Llama-3 8B': 'Llama-3-8B-pt',
    'Gemma-3 1B→Qwen-3 8B': 'Qwen-3-8B-pt',
    'Qwen-3 1.7B→Llama-3 8B': 'Llama-3-8B-pt',
    'Qwen-3 1.7B→Gemma-3 9B': 'Gemma-3-9B-pt',
}



gains = []
for i, row in df_nudging.iterrows():
    pair_gains = []
    base_model = pair_to_base[row['model_pair']]
    # Map to base model (example for Llama-3 1B→3B uses 3B-pt)
    # You need to define this mapping based on your experiment design
    for task in tasks:
        nudged_perf = row[task]  # Performance with guidance
        base_perf = base_data[base_model][task]  # Performance without guidance
        gain = nudged_perf - base_perf  # Calculate improvement
        pair_gains.append(gain)  # ✅ FIXED - append gain, not nudged_perf

    gains.append(pair_gains)

gains_df = pd.DataFrame(gains, 
                       columns=task_names_abb,
                       index=df_nudging['model_pair'])

#plt.figure(figsize=(12, 8))
#sns.heatmap(gains_df, annot=True, fmt='.2f', cmap='RdYlGn', 
#            center=0, cbar_kws={'label': 'Performance'})
#plt.title('Performance Across Model Pairs and Tasks', fontsize=14, fontweight='bold')
#plt.xlabel('Task', fontsize=12)
#plt.ylabel('Model Pair', fontsize=12)


# Abbreviate model names
#gains_df.index = gains_df.index.str.replace('Llama', 'L').replace('Gemma', 'G').replace('Qwen', 'Q')
gains_df.index = gains_df.index.str.replace('Llama', 'L').str.replace('Gemma', 'G').str.replace('Qwen', 'Q')
plt.figure(figsize=(10, 12))  # Taller for readability
sns.heatmap(gains_df, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, cbar=False,  # ← Remove colorbar
            annot_kws={'size': 20})  # ← Bigger numbers in cells
plt.title('Performance Across Model Pairs and Tasks', fontsize=24, fontweight='bold') #18
plt.xlabel('Task', fontsize=24)  # ← Bigger #16
plt.ylabel('Model Pair', fontsize=24)  # ← Bigger #16
plt.xticks(fontsize=24)#12)#14)  # ← Bigger dataset names
plt.yticks(fontsize=24, rotation=0) #18 #12 # ← Bigger, horizontal

plt.tight_layout()
#plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('performance_heatmap.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()


#cross fam vs within fam: Is it statistically worse
# ============= INSERT THE STATISTICAL ANALYSIS HERE =============

import numpy as np
from scipy import stats

# Map each pair to its base model and family type
base_model_groups = {
    'Llama-3-8B-pt': {
        'within': ['Llama-3 1B→8B', 'Llama-3 3B→8B'],
        'cross': ['Gemma-3 1B→Llama-3 8B', 'Qwen-3 1.7B→Llama-3 8B']
    },
    'Gemma-3-9B-pt': {
        'within': ['Gemma-3 1B→9B', 'Gemma-3 4B→9B'],
        'cross': ['Llama-3 1B→Gemma-3 9B', 'Qwen-3 1.7B→Gemma-3 9B']
    },
    'Qwen-3-8B-pt': {
        'within': ['Qwen-3 1.7B→8B', 'Qwen-3 4B→8B'],  # EXCLUDED as outliers
        'cross': ['Llama-3 1B→Qwen-3 8B', 'Gemma-3 1B→Qwen-3 8B']
    }
}

# Calculate performance deltas (nudged - base) for each pair
def get_performance_deltas(pair_name, base_model_name, tasks):
    """Get performance improvements for a model pair across all tasks"""
    pair_row = df_nudging[df_nudging['model_pair'] == pair_name].iloc[0]
    deltas = []
    for task in tasks:
        nudged_perf = pair_row[task]
        base_perf = base_data[base_model_name][task]
        deltas.append(nudged_perf - base_perf)
    return deltas

# Collect deltas for statistical testing
results_by_base = {}
all_within_deltas = []
all_cross_deltas = []

for base_model, pairs in base_model_groups.items():
    # Skip Qwen within-family (outliers)
    if base_model == 'Qwen-3-8B-pt':
        within_pairs = []  # Exclude Qwen within-family
    else:
        within_pairs = pairs['within']
    
    cross_pairs = pairs['cross']
    
    # Get deltas
    within_deltas = []
    for pair in within_pairs:
        deltas = get_performance_deltas(pair, base_model, tasks)
        within_deltas.extend(deltas)
        all_within_deltas.extend(deltas)
    
    cross_deltas = []
    for pair in cross_pairs:
        deltas = get_performance_deltas(pair, base_model, tasks)
        cross_deltas.extend(deltas)
        all_cross_deltas.extend(deltas)
    
    # Per-base-model statistics
    if len(within_deltas) > 0 and len(cross_deltas) > 0:
        t_stat, p_val = stats.ttest_ind(within_deltas, cross_deltas)
        results_by_base[base_model] = {
            'within_mean': np.mean(within_deltas),
            'within_std': np.std(within_deltas),
            'cross_mean': np.mean(cross_deltas),
            'cross_std': np.std(cross_deltas),
            't_stat': t_stat,
            'p_value': p_val,
            'n_within': len(within_deltas),
            'n_cross': len(cross_deltas)
        }

# Overall test (excluding Qwen within-family)
overall_t, overall_p = stats.ttest_ind(all_within_deltas, all_cross_deltas)

# Print results
print("\n" + "="*70)
print("STATISTICAL COMPARISON: Cross-Family vs Within-Family Performance")
print("(Excluding Qwen→Qwen pairs as outliers)")
print("="*70)

print("\n1. PER-BASE-MODEL RESULTS:")
print("-"*70)
for base_model, results in results_by_base.items():
    print(f"\n{base_model}:")
    print(f"  Within-family: μ={results['within_mean']:.3f}, σ={results['within_std']:.3f} (n={results['n_within']})")
    print(f"  Cross-family:  μ={results['cross_mean']:.3f}, σ={results['cross_std']:.3f} (n={results['n_cross']})")
    print(f"  Effect size:   Δ={results['within_mean'] - results['cross_mean']:.3f}")
    print(f"  t-test:        t={results['t_stat']:.3f}, p={results['p_value']:.4f}", end="")
    if results['p_value'] < 0.001:
        print(" ***")
    elif results['p_value'] < 0.01:
        print(" **")
    elif results['p_value'] < 0.05:
        print(" *")
    else:
        print()

print("\n2. OVERALL RESULTS:")
print("-"*70)
print(f"Within-family (n={len(all_within_deltas)}): μ={np.mean(all_within_deltas):.3f}, σ={np.std(all_within_deltas):.3f}")
print(f"Cross-family (n={len(all_cross_deltas)}):  μ={np.mean(all_cross_deltas):.3f}, σ={np.std(all_cross_deltas):.3f}")
print(f"Effect size: Δ={np.mean(all_within_deltas) - np.mean(all_cross_deltas):.3f}")
print(f"t-test: t={overall_t:.3f}, p={overall_p:.4f}", end="")
if overall_p < 0.001:
    print(" ***")
elif overall_p < 0.01:
    print(" **")
elif overall_p < 0.05:
    print(" *")
else:
    print()

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(all_within_deltas)-1)*np.std(all_within_deltas)**2 + 
                       (len(all_cross_deltas)-1)*np.std(all_cross_deltas)**2) / 
                      (len(all_within_deltas) + len(all_cross_deltas) - 2))
cohens_d = (np.mean(all_within_deltas) - np.mean(all_cross_deltas)) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}", end="")
if abs(cohens_d) >= 0.8:
    print(" (large effect)")
elif abs(cohens_d) >= 0.5:
    print(" (medium effect)")
elif abs(cohens_d) >= 0.2:
    print(" (small effect)")
else:
    print(" (negligible effect)")

# ============= VISUALIZATION: Box Plot Comparison =============

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (base_model, pairs) in enumerate(base_model_groups.items()):
    ax = axes[idx]
    
    # Skip Qwen within-family
    if base_model == 'Qwen-3-8B-pt':
        within_data = []
    else:
        within_data = [get_performance_deltas(pair, base_model, tasks) 
                      for pair in pairs['within']]
        within_data = [d for sublist in within_data for d in sublist]  # Flatten
    
    cross_data = [get_performance_deltas(pair, base_model, tasks) 
                 for pair in pairs['cross']]
    cross_data = [d for sublist in cross_data for d in sublist]  # Flatten
    
    # Create box plot
    data_to_plot = []
    labels = []
    if len(within_data) > 0:
        data_to_plot.append(within_data)
        labels.append('Within-Family')
    data_to_plot.append(cross_data)
    labels.append('Cross-Family')
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors[-len(data_to_plot):]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add p-value annotation
    if base_model in results_by_base:
        p_val = results_by_base[base_model]['p_value']
        y_max = max(max(within_data) if within_data else 0, max(cross_data))
        y_pos = y_max + 0.05
        
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        ax.text(1.5, y_pos, f'p={p_val:.4f} {sig_text}', 
               ha='center', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Performance Δ (Nudged - Base)', fontsize=11)
    ax.set_title(base_model.replace('-pt', ''), fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('cross_vs_within_family_controlled.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("INTERPRETATION:")
if overall_p < 0.05:
    print("✓ Cross-family guidance is SIGNIFICANTLY WORSE than within-family")
    print(f"  Average degradation: {np.mean(all_within_deltas) - np.mean(all_cross_deltas):.3f}")
else:
    print("✗ No significant difference detected")
print("="*70)



# ============= VISUALIZATION 4: Task-Specific Degradation Rates =============

# Calculate degradation as: (instruct_performance - nudged_performance) / instruct_performance
# Requires mapping model pairs to their instruct-tuned baseline

fig, ax = plt.subplots(figsize=(12, 6))

# Average degradation per task across all model pairs
task_degradations = []
for task in tasks:
    # Calculate average performance for this task
    avg_perf = df_nudging[task].mean()
    task_degradations.append(avg_perf)

x = np.arange(len(task_names))
bars = ax.bar(x, task_degradations, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

ax.set_xlabel('Task', fontsize=12)
ax.set_ylabel('Average Performance', fontsize=12)
ax.set_title('Task-Specific Performance Under Guidance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(task_names, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('task_specific_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============= VISUALIZATION 5: Comparison Across Gap Sizes =============

fig, ax = plt.subplots(figsize=(14, 6))

# Group by gap size bins
df_nudging['gap_bin'] = pd.cut(df_nudging['gap_ratio'], bins=[0, 3, 5, 10], 
                                labels=['Small (≤3×)', 'Medium (3-5×)', 'Large (>5×)'])

# Average performance per gap bin per task
gap_task_perf = []
for task in tasks:
    gap_means = df_nudging.groupby('gap_bin')[task].mean()
    gap_task_perf.append(gap_means.values)

gap_task_perf = np.array(gap_task_perf).T

x = np.arange(len(task_names))
width = 0.25
gap_labels = ['Small (≤3×)', 'Medium (3-5×)', 'Large (>5×)']
colors = ['#2ecc71', '#f39c12', '#e74c3c']

for i, (gap_label, color) in enumerate(zip(gap_labels, colors)):
    offset = width * (i - 1)
    ax.bar(x + offset, gap_task_perf[i], width, label=gap_label, color=color, alpha=0.8)

ax.set_xlabel('Task', fontsize=12)
ax.set_ylabel('Average Performance', fontsize=12)
ax.set_title('Performance by Capability Gap Size', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(task_names, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('performance_by_gap_size.png', dpi=300, bbox_inches='tight')
plt.show()

# ============= VISUALIZATION 6: Base vs Nudged Comparison =============

# For selected model pairs, compare base-pt, base-it, and nudged
selected_pairs = ['Gemma-3 4B→9B', 'Llama-3 3B→8B', 'Qwen-3 1.7B→8B'] #['Llama-3 1B→8B', 'Gemma-3 1B→9B', 'Qwen-3 1.7B→8B']
selected_indices = [df_nudging[df_nudging['model_pair'] == pair].index[0] 
                   for pair in selected_pairs]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (pair_idx, pair_name) in enumerate(zip(selected_indices, selected_pairs)):
    ax = axes[idx]
    
    # Get performances
    nudged_perfs = [df_nudging.loc[pair_idx, task] for task in tasks]
    base_model_name = pair_to_base[pair_name]
    # Map to base models (you need to define this mapping)
    # Example placeholders
    base_perfs = [base_data[base_model_name][task] for task in tasks] 
    instruct_model_name = base_model_name.replace('-pt', '-it')
    instruct_perfs = [base_data[instruct_model_name][task] for task in tasks]
    
    x = np.arange(len(task_names))
    width = 0.25
    
    ax.bar(x - width, base_perfs, width, label='Base (Pretrained)', color='#95a5a6', alpha=0.8)
    ax.bar(x, instruct_perfs, width, label='Instruction-Tuned', color='#3498db', alpha=0.8)
    ax.bar(x + width, nudged_perfs, width, label='Nudged', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Task', fontsize=10)
    ax.set_ylabel('Performance', fontsize=10)
    ax.set_title(pair_name, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('base_vs_nudged_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("All visualizations generated successfully!")
print("\nKey metrics summary:")
print(f"Average nudging % across all pairs: {df_nudging[[f'{t}_pct' for t in tasks]].mean().mean():.1f}%")
print(f"Correlation between capability gap and performance:")
for task in tasks:
    r, p = pearsonr(df_nudging['capability_gap'], df_nudging[task])
    print(f"  {task}: r={r:.3f}, p={p:.3f}")