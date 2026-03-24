import sinter
import chromobius
import os
import numpy as np
from src.color_code_utils.color_code_circuits.color_code_circuit_666 import ColorCodeCircuit666

def generate_frozen_dataset():
    distances = [5, 7, 9]
    noise_levels = np.logspace(-4, -2, 30)
    
    # 1. Build tasks
    tasks = [
        sinter.Task(
            circuit=ColorCodeCircuit666(d, d*4, p, basis='Z').get_circuit(),
            json_metadata={'d': d, 'p': p, 'r': d*4},
        )
        for d in distances
        for p in noise_levels
    ]

    print("Starting massive data collection. This will take a while...")
    
    # 2. Run and SAVE to CSV
    sinter.collect(
        tasks=tasks,
        num_workers=os.cpu_count(),
        max_shots=50_000_000,   
        max_errors=5000,        
        save_resume_filepath="final_report_666_CC_ZL_data.csv", 
        print_progress=True,    
        decoders=['chromobius'],
        custom_decoders=chromobius.sinter_decoders(),
    )
    print("Data collection complete and saved to final_report_data.csv!")

if __name__ == '__main__':
    generate_frozen_dataset()