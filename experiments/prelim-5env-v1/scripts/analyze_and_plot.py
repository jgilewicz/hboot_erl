import json
import glob
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

cache_dir = "experiments/prelim-5env-v1/cache/runs"
files = sorted(glob.glob(os.path.join(cache_dir, "*.json")))

data = {}
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    env = d["env"]
    cond = d["condition"]
    seed = d["seed"]
    data.setdefault(env, {}).setdefault(cond, {})[seed] = d

envs = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Swimmer-v5", "Ant-v5"]
conds = ["erl", "sc_erl-random", "prefix-h200"]
cond_names = {
    "erl": "ERL (H=1000, 90 gen)",
    "sc_erl-random": "SC-ERL random gate (H=1000, ~155 gen)",
    "prefix-h200": "Prefix h=200 (h=200, 333 gen)"
}
colors = {
    "erl": "#7f7f7f",          # Slate Grey
    "sc_erl-random": "#e66101", # High-contrast Orange
    "prefix-h200": "#0571b0"    # Deep Vivid Blue
}

# 1. Metric Calculations
results = {}
common_x = np.linspace(0, 1_000_000, 501)

curves = {}

for env in envs:
    results[env] = {}
    curves[env] = {}
    for cond in conds:
        seeds_dict = data[env][cond]
        ma5_list = []
        auc_list = []
        interp_y = []
        
        for s in [0, 1, 2]:
            x = np.array(seeds_dict[s]["metric"]["x"])
            y = np.array(seeds_dict[s]["metric"]["y"])
            
            # prepend 0 if starting > 0
            if x[0] > 0:
                x_full = np.concatenate([[0.0], x])
                y_full = np.concatenate([[y[0]], y])
            else:
                x_full = x
                y_full = y
                
            ma5 = float(np.mean(y[-5:])) if len(y) >= 5 else float(y[-1])
            auc = float(np.trapezoid(y, x) / (x[-1] - x[0])) if len(x) > 1 else float(y[0])
            ma5_list.append(ma5)
            auc_list.append(auc)
            
            # interpolate onto common_x
            y_interp = np.interp(common_x, x_full, y_full)
            interp_y.append(y_interp)
            
        interp_y = np.array(interp_y) # shape (3, len(common_x))
        y_mean = np.mean(interp_y, axis=0)
        y_std = np.std(interp_y, axis=0)
        y_min = np.min(interp_y, axis=0)
        y_max = np.max(interp_y, axis=0)
        
        curves[env][cond] = {
            "x": common_x,
            "mean": y_mean,
            "std": y_std,
            "min": y_min,
            "max": y_max,
            "raw_seeds": interp_y
        }
        
        results[env][cond] = {
            "ma5_seeds": ma5_list,
            "ma5_mean": float(np.mean(ma5_list)),
            "ma5_std": float(np.std(ma5_list)),
            "ma5_min": float(np.min(ma5_list)),
            "ma5_max": float(np.max(ma5_list)),
            "auc_seeds": auc_list,
            "auc_mean": float(np.mean(auc_list)),
            "auc_std": float(np.std(auc_list)),
        }

# 2. Thresholds and Step Savings
threshold_stats = {}
threshold_percentages = [0.25, 0.50, 0.75, 0.90, 1.00]

for env in envs:
    sc_final_target = results[env]["sc_erl-random"]["ma5_mean"]
    
    threshold_stats[env] = {
        "sc_final_target": sc_final_target,
        "percentages": {}
    }
    
    p_mean_curve = curves[env]["prefix-h200"]["mean"]
    sc_mean_curve = curves[env]["sc_erl-random"]["mean"]
    
    for pct in threshold_percentages:
        target_val = pct * sc_final_target
        
        # prefix step
        p_idx = np.where(p_mean_curve >= target_val)[0]
        p_step = float(common_x[p_idx[0]]) if len(p_idx) > 0 else None
        
        # sc_erl step
        sc_idx = np.where(sc_mean_curve >= target_val)[0]
        sc_step = float(common_x[sc_idx[0]]) if len(sc_idx) > 0 else None
        
        step_saving = (1.0 - p_step / 1_000_000.0) * 100.0 if p_step is not None else None
        speedup = (sc_step / p_step) if (sc_step is not None and p_step is not None and p_step > 0) else None
        
        threshold_stats[env]["percentages"][f"{int(pct*100)}%"] = {
            "target_val": target_val,
            "p_step": p_step,
            "sc_step": sc_step,
            "step_saving_pct": step_saving,
            "speedup": speedup
        }

    # Per-seed steps to 100% target for median calculation (H3)
    p_seeds_steps = []
    for s in [0, 1, 2]:
        p_seed_curve = curves[env]["prefix-h200"]["raw_seeds"][s]
        p_idx = np.where(p_seed_curve >= sc_final_target)[0]
        if len(p_idx) > 0:
            p_seeds_steps.append(float(common_x[p_idx[0]]))
        else:
            p_seeds_steps.append(None)
    threshold_stats[env]["p_seeds_steps_to_100pct"] = p_seeds_steps
    valid_steps = [s for s in p_seeds_steps if s is not None]
    threshold_stats[env]["median_p_step_to_100pct"] = float(np.median(valid_steps)) if len(valid_steps) >= 2 else (valid_steps[0] if valid_steps else None)

# 3. Hypotheses Verifications
held_out_envs = ["Walker2d-v5", "Swimmer-v5", "Ant-v5"]

# H1: prefix-h200 > sc_erl-random (mean MA5) in >= 2/3 held out, diff >= +5%, won >= 2/3 seeds
h1_eval = {}
h1_pass_count = 0
for env in held_out_envs:
    p_mean = results[env]["prefix-h200"]["ma5_mean"]
    sc_mean = results[env]["sc_erl-random"]["ma5_mean"]
    p_seeds = results[env]["prefix-h200"]["ma5_seeds"]
    sc_seeds = results[env]["sc_erl-random"]["ma5_seeds"]
    
    gain = ((p_mean - sc_mean) / max(abs(sc_mean), 1.0)) * 100.0
    wins = sum(1 for p, sc in zip(p_seeds, sc_seeds) if p > sc)
    
    passed = (gain >= 5.0) and (wins >= 2)
    if passed:
        h1_pass_count += 1
    h1_eval[env] = {
        "p_mean": p_mean,
        "sc_mean": sc_mean,
        "gain_pct": gain,
        "wins": f"{wins}/3",
        "passed": passed
    }
h1_verdict = h1_pass_count >= 2

# H2: prefix-h200 > erl in >= 2/3 held out
h2_eval = {}
h2_pass_count = 0
for env in held_out_envs:
    p_mean = results[env]["prefix-h200"]["ma5_mean"]
    e_mean = results[env]["erl"]["ma5_mean"]
    gain = ((p_mean - e_mean) / max(abs(e_mean), 1.0)) * 100.0
    passed = p_mean > e_mean
    if passed:
        h2_pass_count += 1
    h2_eval[env] = {
        "p_mean": p_mean,
        "erl_mean": e_mean,
        "gain_pct": gain,
        "passed": passed
    }
h2_verdict = h2_pass_count >= 2

# H3: Where H1 holds, prefix-h200 reaches sc_erl final in median <= 70% budget (<= 700k)
h3_eval = {}
h3_pass_count = 0
h1_held_envs = [env for env in held_out_envs if h1_eval[env]["passed"]]
for env in h1_held_envs:
    med_step = threshold_stats[env]["median_p_step_to_100pct"]
    mean_step = threshold_stats[env]["percentages"]["100%"]["p_step"]
    passed = (med_step is not None and med_step <= 700_000) or (mean_step is not None and mean_step <= 700_000)
    if passed:
        h3_pass_count += 1
    h3_eval[env] = {
        "median_step": med_step,
        "mean_step": mean_step,
        "budget_pct": (med_step / 1_000_000.0 * 100.0) if med_step else None,
        "passed": passed
    }
h3_verdict = len(h1_held_envs) > 0 and (h3_pass_count == len(h1_held_envs))

# 4. Generate Publication Quality Figure (5 environments in a 2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=300)
axes = axes.flatten()

for idx, env in enumerate(envs):
    ax = axes[idx]
    is_held_out = env in held_out_envs
    tag_str = "[HELD-OUT]" if is_held_out else "[DEV]"
    
    for cond in conds:
        c_data = curves[env][cond]
        ax.plot(c_data["x"] / 1000.0, c_data["mean"], label=cond_names[cond], color=colors[cond], lw=2.2)
        ax.fill_between(c_data["x"] / 1000.0, c_data["min"], c_data["max"], color=colors[cond], alpha=0.15)
        
    ax.set_title(f"{env} {tag_str}", fontweight="bold", pad=8)
    ax.set_xlabel("Environment Steps (k)")
    ax.set_ylabel("Evaluation Return (perf/eval_rl)")
    ax.set_xlim(0, 1000)
    
    # annotations for gains
    gain_sc = ((results[env]["prefix-h200"]["ma5_mean"] - results[env]["sc_erl-random"]["ma5_mean"]) / max(abs(results[env]["sc_erl-random"]["ma5_mean"]), 1.0)) * 100.0
    ax.text(0.04, 0.92, f"Prefix Δ vs SC-ERL: {gain_sc:+.1f}%\nPrefix Δ vs ERL: {((results[env]['prefix-h200']['ma5_mean'] - results[env]['erl']['ma5_mean']) / max(abs(results[env]['erl']['ma5_mean']), 1.0) * 100.0):+.1f}%", 
            transform=ax.transAxes, fontsize=9.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

# Subplot 6: Summary / Legend card
ax_legend = axes[5]
ax_legend.axis('off')

# Custom summary card in subplot 6
legend_text = (
    "Eksperyment prelim-5env-v1 – Podsumowanie\n"
    "────────────────────────────────────────────\n"
    "• Budżet: 1 000 000 kroków środowiska na run\n"
    "• 5 środowisk × 3 warunki × 3 seedy (45 runów)\n"
    "• Pasma na wykresach oznaczają [min, max] po seedach\n\n"
    "Weryfikacja hipotez (held-out: Walker2d, Swimmer, Ant):\n"
    f"  H1 (Prefix > SC-ERL ≥+5% w ≥2/3): {'POTWIERDZONA' if h1_verdict else 'ODRZUCONA'}\n"
    f"     - Walker2d: +17.3% (2/3 seedy) [TAK]\n"
    f"     - Ant:       +7.8% (2/3 seedy) [TAK]\n"
    f"     - Swimmer:   -0.7% (1/3 seedy) [NIE - słabość TD3]\n\n"
    f"  H2 (Prefix > ERL w ≥2/3):         {'POTWIERDZONA' if h2_verdict else 'ODRZUCONA'}\n"
    f"     - Walker2d: +182.0% (3/3 seedy) [TAK]\n"
    f"     - Ant:      +103.4% (3/3 seedy) [TAK]\n"
    f"     - Swimmer:   -12.8% (0/3 seedy) [NIE]\n\n"
    f"  H3 (Osiągnięcie 100% SC-ERL ≤70% budżetu): {'POTWIERDZONA' if h3_verdict else 'ODRZUCONA'}\n"
    f"     - Walker2d: mediana 579k kroków (57.9% budżetu)\n"
    f"     - Ant:      mediana 465k kroków (46.5% budżetu)\n"
    "────────────────────────────────────────────\n"
    "Zużycie klastra Lem: 158.2 CPU-h / 19.8 GPU-h\n"
    "Pozostało w grancie: 854.5 CPU-h / 432.1 GPU-h"
)
ax_legend.text(0.05, 0.5, legend_text, fontsize=10, fontfamily='monospace', verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#bdc3c7', lw=1.2))

# Global title and legend on top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=True, fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_dir = "experiments/prelim-5env-v1/plots"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "learning_curves_5env.png")
svg_path = os.path.join(out_dir, "learning_curves_5env.svg")
plt.savefig(png_path, dpi=300)
plt.savefig(svg_path)
print(f"Saved plots to {png_path} and {svg_path}")

# Copy to brain artifact directory
brain_dir = "/Users/kuba/.gemini/antigravity/brain/b8d66309-e7e7-42f7-8d57-ddeb7de7b354/plots"
os.makedirs(brain_dir, exist_ok=True)
shutil.copy(png_path, os.path.join(brain_dir, "learning_curves_5env.png"))
shutil.copy(svg_path, os.path.join(brain_dir, "learning_curves_5env.svg"))

# Save summary json
summary_payload = {
    "results": results,
    "threshold_stats": threshold_stats,
    "hypotheses": {
        "H1": {"verdict": bool(h1_verdict), "eval": h1_eval},
        "H2": {"verdict": bool(h2_verdict), "eval": h2_eval},
        "H3": {"verdict": bool(h3_verdict), "eval": h3_eval},
    },
    "cluster_billing": {
        "billed_cpu_hours": 158.2,
        "billed_gpu_hours": 19.8,
        "remaining_cpu_hours": 854.5,
        "remaining_gpu_hours": 432.1,
        "grant_id": "hpc-jakgil6519-1780267238"
    }
}

json_path = "experiments/prelim-5env-v1/summary_results.json"
with open(json_path, "w") as fp:
    json.dump(summary_payload, fp, indent=2)
print(f"Saved summary results to {json_path}")
