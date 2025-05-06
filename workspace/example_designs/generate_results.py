import os
import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt

with open("./example_designs/eyeriss_like/arch.yaml", "r") as f:
    CONFIG = f.readlines()

DATASETS = {
    # "ConvNeXt": "ConvNeXt",
    # "AlexNet": "CONV/AlexNet",
    # "Resnet18": "resnet18",
    "ViT": "vision_transformer",
}

SIZES = {
    "5060": {"__meshX__": 10, "__meshY__": 12, "__datawidth__": 2},
    "5080": {"__meshX__": 28, "__meshY__": 12, "__datawidth__": 4},
    "RTX Pro 6000": {"__meshX__": 64, "__meshY__": 12, "__datawidth__": 16},
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

for accelerator, s in SIZES.items():
    loaded_config = [x for x in CONFIG]
    for i in range(len(loaded_config)):
        for k, v in s.items():
            if k in loaded_config[i]:
                loaded_config[i] = loaded_config[i].replace(k, str(v))
    with open("./example_designs/eyeriss_like/arch.yaml", "w") as f:
        f.writelines(loaded_config)

    for model, model_path in DATASETS.items():
        os.system(f"python3 run_example_designs.py --architecture eyeriss_like --problem {model_path} --n_jobs 2 --clear-outputs")
        time.sleep(1)
        os.system(f"python3 run_example_designs.py --architecture eyeriss_like --problem {model_path} --n_jobs 2")
    
        results = []
        for d in os.listdir("example_designs/eyeriss_like/outputs"):
            layer_name = f"{model}_{d}"
            layer_stats = parse_summary_stats(f"example_designs/eyeriss_like/outputs/{d}/timeloop-mapper.stats.txt")
            layer_stats["layer"] = layer_name
            results.append(layer_stats)
        
        df = pd.DataFrame(results).sort_values("layer")
        df.to_csv(f"{accelerator}_{model}_layer_stats.csv", index=False)

    with open("./example_designs/eyeriss_like/arch.yaml", "w") as f:
        f.writelines(CONFIG)
print("Done")

    