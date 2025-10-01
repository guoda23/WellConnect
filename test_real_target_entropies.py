import os
import pickle

import os
import pickle

def extract_run_number(folder_name):
    """Extracts the run number from a folder named 'experiment_run_42'"""
    try:
        return int(folder_name.split('_')[-1])
    except:
        return -1  # fallback for weird folder names

def check_saved_group_entropies(batch_folder, print_all=False):
    """
    Loops through all saved experiment folders, and prints:
    - Target entropy from params
    - Real entropy per group
    - Δ (difference)
    """
    print(f"\n=== Checking saved group entropies in {batch_folder} ===")

    # Sort subfolders like experiment_run_1, experiment_run_2, ..., experiment_run_120
    sorted_subfolders = sorted(
        [f for f in os.listdir(batch_folder) if f.startswith("experiment_run_")],
        key=extract_run_number
    )

    for subfolder in sorted_subfolders:
        folder_path = os.path.join(batch_folder, subfolder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".pkl"):
                with open(os.path.join(folder_path, file), "rb") as f:
                    data = pickle.load(f)
                    target_entropy = data["params"].get("target_entropy")
                    groups = data["groups"]

                    print(f"\n{subfolder}: Target entropy = {target_entropy:.4f}")
                    for group in groups:
                        real_entropy = getattr(group, "real_entropy", None)
                        if real_entropy is None:
                            print(f"  Group {group.group_id}: ❌ no real_entropy found")
                        else:
                            delta = abs(real_entropy - target_entropy)
                            if print_all or delta > 1e-6:
                                status = "✅" if delta <= 1e-6 else "⚠️"
                                print(f"  Group {group.group_id}: Real = {real_entropy:.4f} | Δ = {delta:.6f} {status}")


check_saved_group_entropies("Experiment_data/homophily_function_retrievability/deterministic/batch_2025-09-26_22-47-38")
