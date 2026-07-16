import os
import sys

# ensure imports resolve whether run from project root or from color_code_experiments/
sys.path.insert(0, os.path.dirname(__file__))

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")

from analysis_utils import (
    load_data,
    load_and_flatten_data,
    calculate_robust_thresholds,
    find_best_crossing_point,
    load_and_filter_stats,
    plot_sinter_thresholds,
    plot_qec_thresholds,
    generate_threshold_plots,
    plot_mapping_comparison,
    plot_mapping_heatmap,
    generate_log_ratio_heatmaps,
    generate_1x3_log_ratio_heatmaps,
    plot_topology_comparison,
    plot_parallelism_and_overhead,
    generate_analytical_dashboard,
    generate_dashboard,
    plot_qec_architecture_comparison,
)


# Section 1: Color Code Threshold Analysis (Sinter CSV data)

def run_xl_threshold_analysis():
    import sinter
    with open("final_report_666_CC_XL_data.csv", "r") as f:
        samples = sinter.stats_from_csv_files(f)

    import matplotlib.pyplot as plt
    calculate_robust_thresholds(samples, estimated_crossing=0.003)
    fig, ax = plt.subplots()
    sinter.plot_error_rate(
        ax=ax, stats=samples,
        x_func=lambda s: s.json_metadata['p'],
        group_func=lambda s: s.json_metadata['d'],
        failure_units_per_shot_func=lambda s: 1,
    )
    ax.loglog()
    ax.set_title("Color Code Threshold Plot (Static Data)")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical X Error Rate ($P_L$)")
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle='--')
    ax.set_xlim(2e-3, 7e-3)
    ax.set_ylim(1e-2, 1)
    ax.legend(title='Distance')
    plt.savefig(os.path.join(PLOTS_DIR, "ImplCCThresholdPlot_Final_XL.png"))
    plt.show()


def run_zl_threshold_analysis():
    import sinter
    import matplotlib.pyplot as plt
    with open("final_report_666_CC_ZL_data.csv", "r") as f:
        samples = sinter.stats_from_csv_files(f)

    calculate_robust_thresholds(samples, estimated_crossing=0.0025)
    fig, ax = plt.subplots()
    sinter.plot_error_rate(
        ax=ax, stats=samples,
        x_func=lambda s: s.json_metadata['p'],
        group_func=lambda s: s.json_metadata['d'],
        failure_units_per_shot_func=lambda s: 1,
    )
    ax.loglog()
    ax.set_title("Color Code Threshold Plot (Static Data)")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Z Error Rate ($P_L$)")
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle='--')
    ax.legend(title='Distance')
    plt.savefig(os.path.join(PLOTS_DIR, "ImplCCThresholdPlot_Final_ZL.png"))
    plt.show()


def run_superdense_threshold_analysis():
    superdense_stats = load_and_filter_stats("stats.csv", filter_keyword="midout_color_code_Z")
    if superdense_stats:
        plot_sinter_thresholds(superdense_stats)
        calculate_robust_thresholds(superdense_stats, estimated_crossing=0.0040, min_d=7)
        find_best_crossing_point(superdense_stats, min_guess=0.004, max_guess=0.007, steps=60, min_d=7)


# Mapping Comparison — Normal vs Sierpinsk

def run_mapping_comparison_single_round():
    cc_data = r".\data\cc_halving_single_round040426.json"
    cc_tri_data = r".\data\cc_bounded_nn_single_round040426.json"
    valid = plot_mapping_comparison(cc_data, cc_tri_data)
    plot_mapping_heatmap(valid)


def run_mapping_comparison_multi_round():
    cc_data = r".\data\cc_halving9326.json"
    cc_tri_data = r".\data\cc_triangular_vec9326.json"
    plot_mapping_comparison(cc_data, cc_tri_data)


def run_surface_vs_color_comparison():
    sc_data = r".\data\surface_halving13326.json"
    cc_tri_data = r".\data\cc_triangular_vec9326.json"
    plot_mapping_comparison(sc_data, cc_tri_data,
                            label_norm="Surface Code", label_tri="Color Code")


# Log-Ratio Heatmaps

def run_heatmaps_dist3_9():
    halving  = r".\data\dist3-9data\cc_halving_single_round_3-9_050426.json"
    unbounded = r".\data\dist3-9data\cc_unbounded_single_round_3-9_050426.json"
    bounded  = r".\data\dist3-9data\cc_bounded_single_round_3-9_050426.json"
    knn      = r".\data\dist3-9data\cc_knn_k=3_single_round_3-9_050426.json"

    generate_log_ratio_heatmaps(halving, unbounded,
                                 label_baseline="Baseline Framework",
                                 label_new="Triangular Unbounded NN")
    generate_log_ratio_heatmaps(halving, knn,
                                 label_baseline="Baseline Framework",
                                 label_new="Triangular k-NN (k=3)")
    generate_log_ratio_heatmaps(halving, bounded,
                                 label_baseline="Baseline Framework",
                                 label_new="Triangular bounded NN")
    generate_1x3_log_ratio_heatmaps(halving, bounded, unbounded, knn,
                                     label_base="Baseline",
                                     labels_alt=["Bounded", "Unbounded", "k-NN"])


# Threshold Plots

def run_488_threshold_plots():
    cc488_switch = r'.\data\tctests\tctestswitch488maybe.json'
    plot_qec_thresholds(cc488_switch, target_capacities=[2, 10, 60])
    generate_threshold_plots(cc488_switch, target_trap_sizes=['2', '10', '60'])


def run_dist3_9_threshold_plots():
    halving = r".\data\dist3-9data\cc_halving_single_round_3-9_050426.json"
    bounded = r".\data\dist3-9data\cc_bounded_single_round_3-9_050426.json"
    target_sizes = ['2', '4', '10', '120']
    generate_threshold_plots(halving, target_trap_sizes=target_sizes)
    generate_threshold_plots(bounded, target_trap_sizes=target_sizes)


# Architecture / Topology Comparison

def run_topology_comparison():
    grid   = r'.\data\topologytestdata\grid_666_d3-15_tc2-10.json'
    linear = r'.\data\topologytestdata\linear_666_d3-11_tc2-30new.json'
    switch = r'.\data\topologytestdata\networked_666_d3-11_tc2-30.json'

    plot_topology_comparison(grid, linear, switch, factor_idx=2)
    plot_parallelism_and_overhead(grid, linear, switch,
                                   save_path=os.path.join(PLOTS_DIR, 'architecture_parallel_routing_effect.png'))


def run_analytical_dashboard():
    grid   = r'.\data\topologytestdata\grid_666_d3-15_tc2-10.json'
    switch = r'.\data\topologytestdata\networked_666_d3-11_tc2-30.json'
    df = load_and_flatten_data(grid, switch, label_1='Grid', label_2='Switch', target_factor_idx=2)
    generate_analytical_dashboard(df)


# QEC Dashboards

def run_666_switch_dashboard():
    generate_dashboard(r'.\data\tctests\tcswitch666.json',
                       savefile=os.path.join(PLOTS_DIR, 'tc_cc666_switch_data.png'))


def run_666_switch_dashboard_with_alt():
    generate_dashboard(
        json_filepath=r'.\data\tctests\tcswitch666.json',
        alt_json_filepath=r'.\data\tctests\tctestdirect_mapped.json',
    )


def run_direct_coord_dashboard():
    generate_dashboard(r'.\data\direct_coord_dist3CCnew.json')


# Surface vs Color Code Comparison

def run_sc_cc_bar_chart():
    sc     = r'.\data\sc_cc_comparison\sc_grid\sc_grid_data.json'
    cc_grid   = r'.\data\sc_cc_comparison\cc_grid_dc\cc_grid_data.json'
    cc_switch = r'.\data\sc_cc_comparison\cc_switch\cc_switch_data.json'
    plot_qec_architecture_comparison(sc, cc_grid, cc_switch)


#uncomment to run specific analyses
if __name__ == "__main__":

    # Threshold Analysis
    # run_xl_threshold_analysis()
    # run_zl_threshold_analysis()
    # run_superdense_threshold_analysis()

    # --- Normal vs Sierpinski Mapping ---
    # run_mapping_comparison_single_round()
    # run_mapping_comparison_multi_round()
    # run_surface_vs_color_comparison()

    # --- Log-Ratio Heatmaps ---
    # run_heatmaps_dist3_9()

    # --- Threshold Plots (JSON) ---
    # run_488_threshold_plots()
    # run_dist3_9_threshold_plots()

    # --- Topology Comparison ---
    # run_topology_comparison()
    # run_analytical_dashboard()

    # --- QEC Dashboard ---
    # run_666_switch_dashboard()
    # run_666_switch_dashboard_with_alt()
    # run_direct_coord_dashboard()

    # ---Surface vs Color Code ---
    # run_sc_cc_bar_chart()

    pass
