"""
Reusable analysis and plotting utilities for QEC simulation data.
"""

import json
import copy
import math
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
from scipy.optimize import curve_fit, OptimizeWarning

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")


# 1. Data Loading

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_and_flatten_data(json_filepath_1, json_filepath_2,
                           label_1='Grid', label_2='Linear',
                           target_factor_idx=2,
                           target_distances=None, target_capacities=None):
    """Parses two nested QEC JSON files into a flat DataFrame for cross-topology analysis."""
    if target_distances is None:
        target_distances = [3, 5, 7, 9, 11]
    if target_capacities is None:
        target_capacities = [2, 5, 10]

    records = []

    def _process(data, topo_name):
        for d_str in data['ElapsedTime']['Forwarding']:
            d = int(d_str)
            for cap_str in data['ElapsedTime']['Forwarding'][d_str]:
                cap = int(cap_str)
                try:
                    elapsed = data['ElapsedTime']['Forwarding'][d_str][cap_str]
                    ops = data['Operations']['Forwarding'][d_str][cap_str]
                    concurrency = data['MeanConcurrency']['Forwarding'][d_str][cap_str]
                    err_list = data['LogicalErrorRates']['Forwarding'].get(d_str, {}).get(cap_str, [])
                    log_err = err_list[target_factor_idx] if len(err_list) > target_factor_idx else None
                    if log_err is not None:
                        records.append({
                            'Topology': topo_name, 'Distance': d, 'Capacity': cap,
                            'Elapsed Time (ms)': elapsed * 1000,
                            'Routing Operations': ops,
                            'Mean Concurrency': concurrency,
                            'Logical Error Rate': log_err,
                        })
                except KeyError:
                    continue

    _process(load_data(json_filepath_1), label_1)
    _process(load_data(json_filepath_2), label_2)

    df = pd.DataFrame(records)
    df = df[df['Distance'].isin(target_distances)]
    df = df[df['Capacity'].isin(target_capacities)]
    return df


# 2. Threshold Fitting

def log_scaling_threshold_model(data, log_c, p_th_star, alpha):
    p, d = data
    return log_c + alpha * ((d + 1) / 2) * np.log(p / p_th_star)


def cross_threshold_model(data, A, B, C, p_th_cross, v_0):
    p, d = data
    x = (p - p_th_cross) * (d ** (1 / v_0))
    return A + B * x + C * (x ** 2)


def calculate_robust_thresholds(samples, estimated_crossing=0.006, min_d=7):
    ps = np.array([s.json_metadata['p'] for s in samples])
    ds = np.array([s.json_metadata['d'] for s in samples])
    shots = np.array([max(s.shots, 1) for s in samples])
    errors = np.array([s.errors for s in samples])

    y = errors / shots
    y_err = np.sqrt(y * (1.0 - y) / shots)
    zero_mask = y == 0
    y[zero_mask] = 1.0 / shots[zero_mask]
    y_err[zero_mask] = 1.0 / shots[zero_mask]

    popt_scale_log, popt_cross = None, None

    print("\n--- Calculating Robust Scaling Threshold ---")
    scale_mask = (ps <= estimated_crossing * 0.6) & (ds >= min_d)
    if np.sum(scale_mask) >= 3:
        log_y = np.log(y[scale_mask])
        log_y_err = y_err[scale_mask] / y[scale_mask]
        try:
            popt_scale_log, pcov = curve_fit(
                log_scaling_threshold_model,
                (ps[scale_mask], ds[scale_mask]), log_y,
                p0=[np.log(0.5), estimated_crossing, 1.0],
                bounds=([-np.inf, 0.0001, 0.1], [10.0, 0.05, 10.0]),
                maxfev=20000, sigma=log_y_err, absolute_sigma=True
            )
            perr = np.sqrt(np.diag(pcov))
            print(f"p*_th (Scaling): {popt_scale_log[1]:.7f} ± {perr[1]:.7f}")
            print(f"c = {np.exp(popt_scale_log[0]):.5f}")
            print(f"alpha = {popt_scale_log[2]:.5f} ± {perr[2]:.5f}")
        except Exception as e:
            print(f"Scaling fit failed: {e}")

    print("\n--- Calculating Robust Cross Threshold ---")
    near_mask = (ps >= estimated_crossing * 0.75) & (ps <= estimated_crossing * 1.25) & (ds >= min_d)
    print(f"Using {np.sum(near_mask)} data points for cross-threshold fit...")
    if np.sum(near_mask) >= 6:
        try:
            popt_cross, pcov = curve_fit(
                cross_threshold_model,
                (ps[near_mask], ds[near_mask]), y[near_mask],
                p0=[np.mean(y[near_mask]), 10.0, 10.0, estimated_crossing, 1.0],
                bounds=([0.0, -np.inf, -np.inf, 0.001, 0.1], [1.0, np.inf, np.inf, 0.05, 5.0]),
                maxfev=20000, sigma=y_err[near_mask], absolute_sigma=True
            )
            perr = np.sqrt(np.diag(pcov))
            print(f"p^x_th (Cross): {popt_cross[3]:.7f} ± {perr[3]:.7f}")
            print(f"v_0 (Crit Exp): {popt_cross[4]:.5f} ± {perr[4]:.5f}")
        except Exception as e:
            print(f"Cross fit failed: {e}")

    return popt_scale_log, popt_cross


def find_best_crossing_point(samples, min_guess=0.003, max_guess=0.008, steps=50, min_d=7):
    """Sweeps crossing-point guesses and returns the popt with lowest threshold uncertainty."""
    print(f"\n--- Sweeping for Optimal Crossing Point ({min_guess} to {max_guess}) ---")

    ps = np.array([s.json_metadata['p'] for s in samples])
    ds = np.array([s.json_metadata['d'] for s in samples])
    shots = np.array([max(s.shots, 1) for s in samples])
    errors = np.array([s.errors for s in samples])
    y = errors / shots
    y_err = np.sqrt(y * (1.0 - y) / shots)
    zero_mask = y == 0
    y[zero_mask] = 1.0 / shots[zero_mask]
    y_err[zero_mask] = 1.0 / shots[zero_mask]

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        for guess in np.linspace(min_guess, max_guess, steps):
            near_mask = (ps >= guess * 0.75) & (ps <= guess * 1.25) & (ds >= min_d)
            if np.sum(near_mask) < 6:
                continue
            try:
                popt, pcov = curve_fit(
                    cross_threshold_model,
                    (ps[near_mask], ds[near_mask]), y[near_mask],
                    p0=[np.mean(y[near_mask]), 10.0, 10.0, guess, 1.0],
                    bounds=([0.0, -np.inf, -np.inf, 0.001, 0.1], [1.0, np.inf, np.inf, 0.05, 5.0]),
                    maxfev=10000, sigma=y_err[near_mask], absolute_sigma=True
                )
                perr = np.sqrt(np.diag(pcov))
                results.append((guess, popt[3], perr[3], popt))
            except RuntimeError:
                pass

    if not results:
        print("Error: No valid fits found across sweep range.")
        return None

    results.sort(key=lambda x: x[2])
    best = results[0]
    print(f"\n--- Sweep Complete ---")
    print(f"Optimal Window Centered At: {best[0]:.6f}")
    print(f"p^x_th (Cross):             {best[1]:.7f} ± {best[2]:.7f}")
    print(f"v_0 (Crit Exp):             {best[3][4]:.5f}")

    guesses = [r[0] for r in results]
    uncertainties = [r[2] for r in results]
    plt.figure(figsize=(8, 4))
    plt.plot(guesses, uncertainties, marker='o', linestyle='-', color='b')
    plt.title("Cross-Threshold Fit Uncertainty Landscape")
    plt.xlabel("Window Center Guess")
    plt.ylabel("Uncertainty in $p^x_{th}$")
    plt.yscale('log')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    return best[3]

# 3. Sinter-based Threshold Plots

def load_and_filter_stats(csv_path, filter_keyword="superdense"):
    """Reads a Sinter CSV and returns TaskStats whose metadata contains filter_keyword."""
    try:
        import sinter
        all_stats = sinter.read_stats_from_csv_files(csv_path)
    except Exception as e:
        print(f"Failed to read Sinter CSV: {e}")
        return []
    filtered = [s for s in all_stats if filter_keyword.lower() in str(s.json_metadata).lower()]
    print(f"Loaded {len(filtered)} '{filter_keyword}' points from {csv_path}.")
    return filtered


def plot_sinter_thresholds(stats, title="Superdense Color Code Threshold"):
    import sinter
    fig, ax = plt.subplots(figsize=(8, 6))
    sinter.plot_error_rate(
        ax=ax, stats=stats,
        x_func=lambda s: s.json_metadata['p'],
        group_func=lambda s: s.json_metadata['d'],
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Physical Error Rate ($p$)', fontsize=12)
    ax.set_ylabel('Logical Error Rate ($p_L$)', fontsize=12)
    ax.grid(which='major', linestyle='-', alpha=0.7)
    ax.grid(which='minor', linestyle=':', alpha=0.4)
    ax.set_xlim(2e-3, 7e-3)
    ax.set_ylim(5e-2, 1)
    ax.legend(title="Code Distance ($d$)", fontsize=10)
    plt.tight_layout()
    plt.show()


# 4. Logical Error Rate Plots (JSON data)

def plot_qec_thresholds(json_path, target_capacities=None, save_path=None):
    data = load_data(json_path)
    log_errs = data.get('LogicalErrorRates', {}).get('Forwarding', {})
    px_errs = data.get('PhysicalXErrorRates', {}).get('Forwarding', {})
    pz_errs = data.get('PhysicalZErrorRates', {}).get('Forwarding', {})
    distances = sorted([int(d) for d in log_errs.keys()])

    if not target_capacities:
        caps = set()
        for d in log_errs.values():
            caps.update(d.keys())
        target_capacities = sorted([int(c) for c in caps])
    else:
        target_capacities = [int(c) for c in target_capacities]

    fig, axes = plt.subplots(1, len(target_capacities),
                              figsize=(6 * len(target_capacities), 5), squeeze=False)
    for idx, cap in enumerate(target_capacities):
        ax = axes[0][idx]
        cap_str = str(cap)
        for d in distances:
            d_str = str(d)
            if cap_str in log_errs.get(d_str, {}):
                y_vals = log_errs[d_str][cap_str]
                x_x = px_errs.get(d_str, {}).get(cap_str, [0] * len(y_vals))
                x_z = pz_errs.get(d_str, {}).get(cap_str, [0] * len(y_vals))
                x_vals = [x + z for x, z in zip(x_x, x_z)]
                pts = sorted(zip(x_vals, y_vals))
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker='o', label=f'Distance {d}')
        ax.set_title(f'Trap Capacity: {cap}', fontsize=14)
        ax.set_xlabel('Total Physical Error Rate ($P_x + P_z$)', fontsize=12)
        ax.set_ylabel('Logical Error Rate ($P_L$)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_threshold_plots(json_filepath, target_trap_sizes=None):
    print(f"Loading data from {json_filepath}...")
    data = load_data(json_filepath)
    logical_rates = data['LogicalErrorRates']['Forwarding']
    phys_x_rates = data['PhysicalXErrorRates']['Forwarding']
    phys_z_rates = data['PhysicalZErrorRates']['Forwarding']
    distances = sorted([int(d) for d in logical_rates.keys()])

    common_caps = set(logical_rates[str(distances[0])].keys())
    for d in distances[1:]:
        common_caps &= set(logical_rates[str(d)].keys())
    sorted_caps = sorted(list(common_caps), key=int)
    print(f"Common trap sizes: {sorted_caps}")

    if target_trap_sizes is None:
        target_trap_sizes = ['4', '15', '60']
    target_trap_sizes = [t for t in target_trap_sizes if t in sorted_caps]

    fig, axes = plt.subplots(1, len(target_trap_sizes),
                              figsize=(6 * len(target_trap_sizes), 6), squeeze=False)
    d_markers = {3: 'o', 5: 's', 7: '^'}
    d_colors = {3: 'blue', 5: 'orange', 7: 'green'}

    for i, trap in enumerate(target_trap_sizes):
        ax = axes[0][i]
        for d in distances:
            d_str = str(d)
            y_vals = logical_rates[d_str][trap]
            x_avg = [(x + z) / 2.0 for x, z in
                     zip(phys_x_rates[d_str][trap], phys_z_rates[d_str][trap])]
            ax.plot(x_avg, y_vals,
                    marker=d_markers.get(d, 'x'), color=d_colors.get(d, 'black'),
                    linestyle='-', linewidth=2, markersize=8, label=f'Distance {d}')
        ax.set_title(f'Threshold Plot (Capacity: {trap})', fontsize=14)
        ax.set_xlabel('Total Physical Error Rate ($p_x + p_z$)/2', fontsize=12)
        ax.set_ylabel('Logical Error Rate ($P_L$)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_mapping_comparison(cc_data_path, cc_tri_data_path,
                             label_norm="Normal mapping", label_tri="Sierpinski mapping",
                             shots=1_000_000):
    """
    Plots logical error rate vs physical error rate for two qubit-to-ion mappings.
    Returns the merged DataFrame with proportional differences for use in plot_mapping_heatmap.
    """
    raw = load_data(cc_data_path)
    raw_tri = load_data(cc_tri_data_path)
    label = 'Forwarding'

    def _flatten(raw_data):
        rows = []
        log_d = raw_data['LogicalErrorRates'][label]
        px_d = raw_data['PhysicalXErrorRates'][label]
        pz_d = raw_data['PhysicalZErrorRates'][label]
        for d_str, cap_dict in log_d.items():
            d = int(d_str)
            for cap_str, log_err_list in cap_dict.items():
                cap = int(cap_str)
                for i, log_err in enumerate(log_err_list):
                    px = px_d[d_str][cap_str][i]
                    pz = pz_d[d_str][cap_str][i]
                    rows.append({
                        'd': d, 'capacity': cap, 'point_idx': i,
                        'p': (px + pz) / 2,
                        'logical_error_rate': log_err,
                        'shots': shots,
                    })
        return rows

    df = pd.DataFrame(_flatten(raw))
    df_tri = pd.DataFrame(_flatten(raw_tri))

    plt.figure(figsize=(10, 7))
    for data, name, ls, marker in [
        (df, label_norm, '-', 'o'),
        (df_tri, label_tri, '--', 's'),
    ]:
        for d in sorted(data['d'].unique()):
            subset = data[data['d'] == d].sort_values('p')
            subset = subset[subset['logical_error_rate'] > 0].copy()
            subset['std_err'] = np.sqrt(
                subset['logical_error_rate'] * (1 - subset['logical_error_rate']) / subset['shots']
            )
            plt.errorbar(subset['p'], subset['logical_error_rate'], yerr=subset['std_err'],
                         label=f'{name}, d={d}', linestyle=ls, marker=marker, capsize=3)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate ($p$)')
    plt.ylabel('Logical Error Rate ($P_L$)')
    plt.title('Logical Error Rate vs Distance')
    plt.grid(True, which="both", alpha=0.2)
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k-', alpha=0.25, label='Break-even')
    plt.legend(ncol=2)
    plt.show()

    merged = pd.merge(df, df_tri, on=['d', 'capacity', 'point_idx'], suffixes=('_norm', '_tri'))
    valid = merged[merged['logical_error_rate_norm'] > 0].copy()
    valid['prop_diff'] = (
        (valid['logical_error_rate_tri'] - valid['logical_error_rate_norm'])
        / valid['logical_error_rate_norm']
    )
    print(f"Average proportional difference: {valid['prop_diff'].mean() * 100:.2f}%")
    print(f"Average absolute proportional difference: {valid['prop_diff'].abs().mean() * 100:.2f}%")
    return valid


def plot_mapping_heatmap(valid_merged):
    """Heatmap of Sierpinski vs Normal proportional error rate difference."""
    summary = valid_merged.groupby(['d', 'capacity'])['prop_diff'].mean().reset_index()
    summary['prop_diff_%'] = summary['prop_diff'] * 100
    pivot = summary.pivot(index='capacity', columns='d', values='prop_diff_%')

    print("Average Proportional Difference (%) — Negative (Blue) = Sierpinski better:")
    print(pivot.round(2))

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="vlag", center=0,
                cbar_kws={'label': 'Proportional Difference (%)'}, linewidths=0.5)
    plt.title("Sierpinski vs Normal: Performance Gap Heatmap")
    plt.xlabel("Code Distance ($d$)")
    plt.ylabel("Capacity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# 5. Log-Ratio Heatmaps

def get_global_keys(datasets):
    all_distances, all_capacities = set(), set()
    for data in datasets:
        all_distances.update(data.keys())
        for dist_dict in data.values():
            all_capacities.update(dist_dict.keys())
    distances = sorted([int(d) for d in all_distances])
    capacities = [str(c) for c in sorted([int(c) for c in all_capacities], reverse=True)]
    return distances, capacities


def _build_log_ratio_df(d_base, d_new, distances, capacities, factor_idx):
    rows = []
    for cap in capacities:
        row = []
        for dist in distances:
            ds = str(dist)
            rb = d_base.get(ds, {}).get(cap)
            rn = d_new.get(ds, {}).get(cap)
            if (rb and rn and len(rb) > factor_idx and len(rn) > factor_idx
                    and rb[factor_idx] > 0 and rn[factor_idx] > 0):
                row.append(np.log10(rn[factor_idx] / rb[factor_idx]))
            else:
                row.append(np.nan)
        rows.append(row)
    return pd.DataFrame(rows, index=capacities, columns=distances)


def generate_log_ratio_heatmaps(json_baseline, json_new,
                                 label_baseline="Normal", label_new="Sierpinski"):
    d_base = load_data(json_baseline)['LogicalErrorRates']['Forwarding']
    d_new = load_data(json_new)['LogicalErrorRates']['Forwarding']
    distances, capacities = get_global_keys([d_base, d_new])
    n_factors = max((len(c) for d in d_base.values() for c in d.values()), default=0)
    gate_labels = [1.0, 5.0, 10.0]

    for factor_idx in range(n_factors):
        df = _build_log_ratio_df(d_base, d_new, distances, capacities, factor_idx)

        annot = np.empty_like(df.values, dtype=object)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iloc[i, j]
                annot[i, j] = "NaN" if pd.isna(val) else f"{10 ** (-val):.2f}x"

        limit = min(df.abs().max().max(), 2.0)
        max_tick = math.ceil(limit)
        cbar_ticks = np.arange(-max_tick, max_tick + 1, 1.0)

        cmap = copy.copy(plt.get_cmap("vlag"))
        cmap.set_bad("darkgray")

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(df, annot=annot, fmt="", cmap=cmap, center=0,
                         vmin=-limit, vmax=limit,
                         cbar_kws={'label': 'Relative Improvement Multiplier', 'ticks': cbar_ticks},
                         linewidths=.5)
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels([
            f"{10**(-t):.0f}x" if 10**(-t) >= 1 else f"{10**(-t):g}x"
            for t in cbar_ticks
        ])
        gate_label = gate_labels[factor_idx] if factor_idx < len(gate_labels) else factor_idx
        plt.title(f'{label_new} vs {label_baseline}: Relative Improvement (Gate {gate_label}x)',
                  fontsize=13, pad=15)
        plt.xlabel('Code Distance ($d$)', fontsize=12)
        plt.ylabel('Capacity', fontsize=12)
        plt.tight_layout()

        if factor_idx == 1:
            fname = os.path.join(PLOTS_DIR, f'heatmap_improvement_factor_{gate_label}_{label_new.replace(" ", "_")}.png')
            plt.savefig(fname, dpi=300)
            print(f"Saved: {fname}")
        plt.show()


def generate_1x3_log_ratio_heatmaps(json_baseline, json_alt1, json_alt2, json_alt3,
                                     label_base="Baseline", labels_alt=None):
    if labels_alt is None:
        labels_alt = ["Alt 1", "Alt 2", "Alt 3"]

    d_base = load_data(json_baseline)['LogicalErrorRates']['Forwarding']
    alts = [load_data(p)['LogicalErrorRates']['Forwarding']
            for p in (json_alt1, json_alt2, json_alt3)]

    distances, capacities = get_global_keys([d_base] + alts)
    n_factors = max((len(c) for d in d_base.values() for c in d.values()), default=0)
    gate_labels = [1.0, 5.0, 10.0]

    for factor_idx in range(n_factors):
        dfs, annots = [], []
        for alt in alts:
            df = _build_log_ratio_df(d_base, alt, distances, capacities, factor_idx)
            dfs.append(df)
            ann = np.empty_like(df.values, dtype=object)
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    v = df.iloc[i, j]
                    ann[i, j] = "NaN" if pd.isna(v) else f"{v:.2f}"
            annots.append(ann)

        limit = min(max((df.abs().max().max() for df in dfs), default=1.0), 2.0)
        if pd.isna(limit):
            limit = 1.0

        cmap = copy.copy(plt.get_cmap("vlag"))
        cmap.set_bad("darkgray")

        fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
        for i, ax in enumerate(axes):
            sns.heatmap(dfs[i], ax=ax, annot=annots[i], fmt="", cmap=cmap,
                        center=0, vmin=-limit, vmax=limit,
                        cbar=(i == 2),
                        cbar_kws={'label': 'Log10 Ratio'} if i == 2 else None,
                        linewidths=.5)
            ax.set_title(f'{labels_alt[i]} vs {label_base}', fontsize=14, pad=10)
            ax.set_xlabel('Code Distance ($d$)', fontsize=12)
            if i == 0:
                ax.set_ylabel('Capacity', fontsize=12)

        gate_label = gate_labels[factor_idx] if factor_idx < len(gate_labels) else factor_idx
        fig.suptitle(f'Comparative QEC Mapping Performance (Gate {gate_label}x)',
                     fontsize=16, y=1.02)
        plt.tight_layout()
        if factor_idx == 1:
            fname = os.path.join(PLOTS_DIR, f'heatmap_1x3_logratio_factor_gate_imprv_{gate_label}.png')
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            print(f"Saved: {fname}")
        plt.show()


# 6. Architecture / Topology Comparison

def extract_metric(data, metric, distances, capacities, factor_index=None):
    """Extracts a named metric from QEC JSON data per distance and capacity."""
    extracted = {cap: {'d': [], 'val': []} for cap in capacities}
    for d in distances:
        d_str = str(d)
        if d_str not in data[metric]['Forwarding']:
            continue
        for cap in capacities:
            if cap not in data[metric]['Forwarding'][d_str]:
                continue
            val = data[metric]['Forwarding'][d_str][cap]
            if factor_index is not None:
                if isinstance(val, list) and len(val) > factor_index:
                    val = val[factor_index]
                else:
                    continue
            extracted[cap]['d'].append(d)
            extracted[cap]['val'].append(val)
    return extracted


def plot_topology_comparison(grid_file, linear_file, switch_file,
                              target_capacities=None, target_distances=None,
                              factor_idx=2, extrapolate_distance=16, save_path=None):
    """Two-panel: (a) QEC round time, (b) logical error rate with extrapolation."""
    if target_capacities is None:
        target_capacities = ['2', '5', '10']
    if target_distances is None:
        target_distances = [3, 5, 7, 9, 11]

    colors = {'2': 'navy', '5': 'purple', '10': 'tomato'}
    markers = {'linear': 'x', 'grid': '.', 'switch': '+'}
    linestyles = {'linear': '--', 'grid': '-.', 'switch': ':'}

    topos = {
        'grid': load_data(grid_file),
        'linear': load_data(linear_file),
        'switch': load_data(switch_file),
    }
    times = {t: extract_metric(d, 'ElapsedTime', target_distances, target_capacities)
             for t, d in topos.items()}
    errors = {t: extract_metric(d, 'LogicalErrorRates', target_distances, target_capacities, factor_idx)
              for t, d in topos.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for topo in ('linear', 'grid', 'switch'):
        for cap in target_capacities:
            vals_ms = [v * 1000 for v in times[topo][cap]['val']]
            if vals_ms:
                ax1.plot(times[topo][cap]['d'], vals_ms, color=colors[cap],
                         marker=markers[topo], linestyle=linestyles[topo],
                         label=f'c={cap}, {topo}', alpha=0.7)

            d_vals = np.array(errors[topo][cap]['d'])
            p_vals = np.array(errors[topo][cap]['val'])
            if len(d_vals) > 0:
                ax2.scatter(d_vals, p_vals, color=colors[cap], marker=markers[topo], alpha=0.8,
                            label=f'c={cap}, {topo}')
                valid = p_vals > 0
                if sum(valid) >= 2:
                    m, c = np.polyfit(d_vals[valid], np.log10(p_vals[valid]), 1)
                    ext = np.linspace(min(d_vals), extrapolate_distance, 50)
                    ax2.plot(ext, 10**(m * ext + c), color=colors[cap],
                             linestyle=linestyles[topo], alpha=0.4)

    ax1.set_title('(a) Topology Effect on QEC Round Time', fontsize=14, pad=10)
    ax1.set_xlabel('Distance', fontsize=12)
    ax1.set_ylabel('QEC Round Time (ms)', fontsize=12)
    ax1.set_xticks(target_distances)
    ax1.grid(True, linestyle='-', alpha=0.6)
    ax1.legend()

    ax2.set_title('(b) Topology Effect on Logical Error Rate', fontsize=14, pad=10)
    ax2.set_xlabel('Distance', fontsize=12)
    ax2.set_ylabel('Logical Error Rate', fontsize=12)
    ax2.set_xlim(2, extrapolate_distance)
    ax2.set_xticks(np.arange(2, extrapolate_distance + 1, 2))
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='-', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_parallelism_and_overhead(grid_file, linear_file, switch_file,
                                   target_capacities=None, target_distances=None,
                                   save_path=None):
    """Two-panel: (a) mean concurrency, (b) routing overhead ratio."""
    if target_capacities is None:
        target_capacities = ['2', '5', '10']
    if target_distances is None:
        target_distances = [3, 5, 7, 9, 11]

    colors = {'2': 'navy', '5': 'purple', '10': 'tomato'}
    markers = {'linear': 'x', 'grid': '.', 'switch': '+'}
    linestyles = {'linear': '--', 'grid': '-.', 'switch': ':'}

    topos = {
        'grid': load_data(grid_file),
        'linear': load_data(linear_file),
        'switch': load_data(switch_file),
    }

    def _overhead(data):
        ops = extract_metric(data, 'Operations', target_distances, target_capacities)
        q_ops = extract_metric(data, 'QubitOperations', target_distances, target_capacities)
        ratio = {cap: {'d': [], 'val': []} for cap in target_capacities}
        for cap in target_capacities:
            for d, op, qop in zip(ops[cap]['d'], ops[cap]['val'], q_ops[cap]['val']):
                if qop > 0:
                    ratio[cap]['d'].append(d)
                    ratio[cap]['val'].append(op / qop)
        return ratio

    conc = {t: extract_metric(d, 'MeanConcurrency', target_distances, target_capacities)
            for t, d in topos.items()}
    ratios = {t: _overhead(d) for t, d in topos.items()}

    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

    for topo in ('linear', 'grid', 'switch'):
        for cap in target_capacities:
            for ax, metric in [(ax3, conc[topo]), (ax4, ratios[topo])]:
                cap_data = metric.get(cap, {'d': [], 'val': []})
                if cap_data['d']:
                    ax.plot(cap_data['d'], cap_data['val'], color=colors[cap],
                            marker=markers[topo], linestyle=linestyles[topo],
                            label=f'c={cap}, {topo}', alpha=0.7, linewidth=2, markersize=8)

    ax3.set_title('(a) Parallelisation vs Code Distance', fontsize=14, pad=10)
    ax3.set_xlabel('Code Distance ($d$)', fontsize=12)
    ax3.set_ylabel('Mean Concurrency', fontsize=12)
    ax3.set_xticks(target_distances)
    ax3.grid(True, linestyle='-', alpha=0.6)
    ax3.legend()

    ax4.axhline(y=1.0, color='black', linestyle='-', linewidth=2.5,
                label='Theoretical Baseline (Ratio = 1.0)')
    ax4.set_title('(b) Routing Overhead Ratio vs Code Distance', fontsize=14, pad=10)
    ax4.set_xlabel('Code Distance ($d$)', fontsize=12)
    ax4.set_ylabel('Overhead Multiplier (Total Ops / Qubit Ops)', fontsize=12)
    ax4.set_xticks(target_distances)
    ax4.set_ylim(0, 12)
    ax4.grid(True, which='both', linestyle='-', alpha=0.6)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Topology & Capacity")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_analytical_dashboard(df):
    """2x2 seaborn dashboard: time, concurrency, routing ops, and time-vs-accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.set_theme(style="whitegrid")

    kwargs = dict(hue='Capacity', style='Topology', markers=True, palette='Set1', linewidth=2, markersize=8)
    sns.lineplot(data=df, x='Distance', y='Elapsed Time (ms)', ax=axes[0, 0], **kwargs)
    axes[0, 0].set_title('1. QEC Round Time', fontsize=14)

    sns.lineplot(data=df, x='Distance', y='Mean Concurrency', ax=axes[0, 1], **kwargs)
    axes[0, 1].set_title('2. Parallelization', fontsize=14)

    sns.lineplot(data=df, x='Distance', y='Routing Operations', ax=axes[1, 0], **kwargs)
    axes[1, 0].set_title('3. Routing Overhead', fontsize=14)

    sns.scatterplot(data=df, x='Elapsed Time (ms)', y='Logical Error Rate',
                    hue='Capacity', style='Topology', size='Distance',
                    sizes=(50, 200), palette='Set1', ax=axes[1, 1], alpha=0.8)
    axes[1, 1].set_title('4. Time vs. Accuracy Trade-off', fontsize=14)
    axes[1, 1].set_yscale('log')

    plt.suptitle('QEC Mapping Architecture Analysis (Gate Improvement: 5.0x)', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()


def generate_dashboard(json_filepath, alt_json_filepath=None, savefile=None,
                        time_capacities=None, error_capacities=None,
                        improvements=None, max_distance=20):
    """
    2x2 dashboard: (a) round time vs distance, (b/c/d) extrapolated logical error rates
    at different gate improvement factors.
    alt_json_filepath optionally overlays a second c=2 mapping on all panels.
    """
    if time_capacities is None:
        time_capacities = [2, 3, 5, 10, 15, 60]
    if error_capacities is None:
        error_capacities = [2, 10, 15, 60]
    if improvements is None:
        improvements = [(0, '1x'), (2, '10x'), (3, '15x')]

    y_min, y_max = 1e-9, 1e0
    alt_color = 'teal'
    alt_label = 'c=2 (Alt Mapping)'

    data = load_data(json_filepath)
    times = data.get('ElapsedTime', {}).get('Forwarding', {})
    log_errs = data.get('LogicalErrorRates', {}).get('Forwarding', {})
    distances = sorted([int(d) for d in log_errs.keys()])

    alt_times, alt_log_errs = {}, {}
    if alt_json_filepath:
        alt_data = load_data(alt_json_filepath)
        alt_times = alt_data.get('ElapsedTime', {}).get('Forwarding', {})
        alt_log_errs = alt_data.get('LogicalErrorRates', {}).get('Forwarding', {})

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    time_colors = cm.magma(np.linspace(0.2, 0.8, len(time_capacities)))
    extrap_colors = ['navy', '#A77BB5', '#EAA78C', 'hotpink', 'green']

    # Panel (a): round time
    ax_a = axes[0, 0]
    for i, cap in enumerate(time_capacities):
        cap_str = str(cap)
        xs = [d for d in distances if cap_str in times.get(str(d), {})]
        ys = [times[str(d)][cap_str] * 1000 for d in xs]
        if xs:
            ax_a.plot(xs, ys, color=time_colors[i], marker='.', linewidth=1.5,
                      markersize=6, label=f'c={cap}', alpha=0.8)
    if alt_times:
        xs = [d for d in distances if '2' in alt_times.get(str(d), {})]
        ys = [alt_times[str(d)]['2'] * 1000 for d in xs]
        if xs:
            ax_a.plot(xs, ys, color=alt_color, marker='*', linestyle='--',
                      linewidth=1.5, markersize=8, label=alt_label, alpha=0.9)
    dist_arr = np.array(distances)
    ax_a.plot(dist_arr, [5] * len(dist_arr), 'k--', linewidth=1, label='Lower Bound')
    ax_a.plot(dist_arr, dist_arr ** 2.1, 'k-', linewidth=1, label='Upper Bound')
    ax_a.set_title('(a) Effect of Code Distance on Round Time', fontsize=16, pad=10)
    ax_a.set_xlabel('Distance', fontsize=12)
    ax_a.set_ylabel('Time (ms)', fontsize=12)
    ax_a.set_xticks(np.arange(2, max(distances) + 2, 1))
    ax_a.grid(True, linestyle='-', alpha=0.5)
    ax_a.legend(loc='upper left', fontsize=11)

    # Panels (b), (c), (d): extrapolated error rates
    extrap_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    panel_labels = ['(b)', '(c)', '(d)']

    for ax_idx, (factor_idx, label_mult) in enumerate(improvements):
        ax = extrap_axes[ax_idx]

        for c_idx, cap in enumerate(error_capacities):
            cap_str = str(cap)
            c_color = extrap_colors[c_idx % len(extrap_colors)]
            xs = [d for d in distances
                  if cap_str in log_errs.get(str(d), {})
                  and len(log_errs[str(d)][cap_str]) > factor_idx
                  and log_errs[str(d)][cap_str][factor_idx] > 0]
            ys = [log_errs[str(d)][cap_str][factor_idx] for d in xs]
            if xs:
                xs, ys = np.array(xs), np.array(ys)
                ax.scatter(xs, ys, color=c_color, s=20, alpha=0.9, zorder=3)
                if len(xs) >= 2:
                    log_y = np.log10(ys)
                    m, c = np.polyfit(xs, log_y, 1)
                    ext = np.linspace(0.5, max_distance, 50)
                    ax.plot(ext, 10**(m * ext + c), color=c_color, linestyle='--', linewidth=1.2, zorder=4)
                    std_err = max(np.std(log_y - (m * xs + c)), 0.25)
                    ax.fill_between(ext, 10**(m * ext + c - std_err), 10**(m * ext + c + std_err),
                                    color=c_color, alpha=0.4, edgecolor='none', zorder=2)

        if alt_log_errs:
            xs = [d for d in distances
                  if '2' in alt_log_errs.get(str(d), {})
                  and len(alt_log_errs[str(d)]['2']) > factor_idx
                  and alt_log_errs[str(d)]['2'][factor_idx] > 0]
            ys = [alt_log_errs[str(d)]['2'][factor_idx] for d in xs]
            if xs:
                xs, ys = np.array(xs), np.array(ys)
                ax.scatter(xs, ys, color=alt_color, marker='*', s=50, alpha=0.9, zorder=3)
                if len(xs) >= 2:
                    log_y = np.log10(ys)
                    m, c = np.polyfit(xs, log_y, 1)
                    ext = np.linspace(0.5, max_distance, 50)
                    ax.plot(ext, 10**(m * ext + c), color=alt_color, linestyle=':', linewidth=1.5, zorder=4)
                    std_err = max(np.std(log_y - (m * xs + c)), 0.25)
                    ax.fill_between(ext, 10**(m * ext + c - std_err), 10**(m * ext + c + std_err),
                                    color=alt_color, alpha=0.2, edgecolor='none', zorder=2)

        ax.set_title(f'{panel_labels[ax_idx]} Distance Needed at {label_mult} Improvement',
                     fontsize=16, pad=10)
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Logical Error Rate', fontsize=12)
        ax.set_yscale('log')
        ax.set_xlim(0.5, max_distance)
        if factor_idx != 0:
            ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(2, max_distance + 1, 2))
        ax.grid(True, which='major', linestyle='-', alpha=0.5)
        handles = [mpatches.Patch(color=extrap_colors[i % len(extrap_colors)], alpha=0.5, label=f'c={cap}')
                   for i, cap in enumerate(error_capacities)]
        if alt_log_errs:
            handles.append(mpatches.Patch(color=alt_color, alpha=0.5, label=alt_label))
        loc = 'lower left' if ax_idx == 1 else 'upper right'
        ax.legend(handles=handles, loc=loc, fontsize=11, framealpha=0.9)

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()


def plot_qec_architecture_comparison(sc_grid_file, cc_grid_file, cc_switch_file,
                                      target_distances=None, factor_idx=1, factor_label="5.0x",
                                      trap_capacity='2', save_path=os.path.join(PLOTS_DIR, 'sc_cc_comparison_barchart.png')):
    """Bar chart: surface code vs color code (grid + switch), round time and logical error rate."""
    if target_distances is None:
        target_distances = [3, 5, 7, 9]

    sc_grid = load_data(sc_grid_file)
    cc_grid = load_data(cc_grid_file)
    cc_switch = load_data(cc_switch_file)

    labels = ['Surface Code (Grid)', 'Color Code (Grid)', 'Color Code (Switch)']
    colors = ['#4A90E2', '#50E3C2', '#F5A623']

    def _extract(data_dict, metric, f_idx=None):
        vals = []
        for d in target_distances:
            try:
                val = data_dict[metric]['Forwarding'][str(d)][trap_capacity]
                if f_idx is not None:
                    val = val[f_idx]
                vals.append(val if val > 0 else 1e-10)
            except (KeyError, IndexError):
                vals.append(np.nan)
        return vals

    times = [np.array(_extract(src, 'ElapsedTime')) * 1000
             for src in (sc_grid, cc_grid, cc_switch)]
    lers = [_extract(src, 'LogicalErrorRates', factor_idx)
            for src in (sc_grid, cc_grid, cc_switch)]

    x = np.arange(len(target_distances))
    width = 0.25
    offsets = [-width, 0, width]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for t_vals, label, color, off in zip(times, labels, colors, offsets):
        ax1.bar(x + off, t_vals, width, label=label, color=color, edgecolor='black', alpha=0.85)
    ax1.set_title('QEC Round Time Comparison', fontsize=14, pad=10)
    ax1.set_xlabel('Code Distance ($d$)', fontsize=12)
    ax1.set_ylabel('Elapsed Time (ms)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(target_distances)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)

    for l_vals, label, color, off in zip(lers, labels, colors, offsets):
        ax2.bar(x + off, l_vals, width, label=label, color=color, edgecolor='black', alpha=0.85)
    ax2.set_title(f'Logical Error Rate Comparison (Gate Improvement: {factor_label})',
                  fontsize=14, pad=10)
    ax2.set_xlabel('Code Distance ($d$)', fontsize=12)
    ax2.set_ylabel('Logical Error Rate', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(target_distances)
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-7)
    ax2.grid(axis='y', which='both', linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
