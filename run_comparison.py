from main import main

CONFIGS = [
    "configs/config_comparison_sc_grid.yml",
    "configs/config_comparison_cc_grid_dc.yml",
    "configs/config_comparison_cc_switch.yml"
]

if __name__ == "__main__":
    for config in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running: {config}")
        print('='*60)
        main(config)
    print("\nAll comparison experiments complete.")
