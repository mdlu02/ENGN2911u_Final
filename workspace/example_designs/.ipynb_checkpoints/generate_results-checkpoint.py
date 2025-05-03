import os
import pandas as pd
import matplotlib.pyplot as plt


DATASETS = {
    "ConvNeXt": "ConvNeXt",
    "AlexNet": "CONV/AlexNet",
    "Resnet18": "resnet18",
}

def parse_summary_stats(file_path):
    summary = {}
    computes_section = False
    fJ_per_compute = {}

    with open(file_path, 'r') as f:
        found = False
        for line in f:
            line = line.strip()

            # Skip empty lines
            if line.strip() == "Summary Stats":
                found = True
            if not line or not found:
                continue

            if line.startswith("GFLOPs"):
                summary["GFLOPs"] = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Utilization"):
                summary["Utilization (%)"] = float(line.split(":")[1].strip().replace('%', ''))
            elif line.startswith("Cycles"):
                summary["Cycles"] = int(line.split(":")[1].strip())
            elif line.startswith("Energy"):
                summary["Energy (uJ)"] = float(line.split(":")[1].strip().replace('uJ', ''))
            elif line.startswith("EDP"):
                summary["EDP (J*cycle)"] = float(line.split(":")[1].strip())
            elif line.startswith("Area"):
                summary["Area (mm^2)"] = float(line.split(":")[1].strip().replace('mm^2', ''))
            elif line.startswith("Computes ="):
                summary["Computes"] = int(line.split("=")[1].strip())
                computes_section = True
            elif computes_section and '=' in line:
                key, val = map(str.strip, line.split('='))
                fJ_per_compute[key] = float(val)

    summary["fJ/Compute"] = fJ_per_compute
    return summary


for model, model_path in DATASETS.items():
    os.system(f"python3 run_example_designs.py --architecture eyeriss_like --problem {model_path} --n_jobs 6 --clear-outputs")
    os.system(f"python3 run_example_designs.py --architecture eyeriss_like --problem {model_path} --n_jobs 6")

    results = []
    for d in os.listdir("example_designs/eyeriss_like/outputs"):
        layer_name = f"{model}_{d}"
        layer_stats = parse_summary_stats(f"example_designs/eyeriss_like/outputs/{d}/timeloop-mapper.stats.txt")
        layer_stats["layer"] = layer_name
        results.append(layer_stats)
    
    df = pd.DataFrame(results).sort_values("layer")
    df.to_csv(f"{model}_layer_stats.csv", index=False)

print("Done")

    