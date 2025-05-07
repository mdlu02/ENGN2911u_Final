import os
import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt

os.system("rm -rf ./outputs; mkdir ./outputs")

with open("./example_designs/eyeriss_like/arch_base.yaml", "r") as f:
    CONFIG = f.readlines()

DATASETS = {
    "ConvNeXt": "ConvNeXt",
    "AlexNet": "CONV/AlexNet",
    "Resnet18": "resnet18",
    "ViT": "vision_transformer",
}

SIZES = {
    "5060": {"__meshX__": 10, "__meshY__": 12, "__datawidth__": 2},
    "5080": {"__meshX__": 28, "__meshY__": 12, "__datawidth__": 4},
    "RTX_Pro_6000": {"__meshX__": 64, "__meshY__": 12, "__datawidth__": 16},
}


def parse_summary_stats(file_path):
    summary = {}
    intensity_section = False
    intensity_sub = None
    summary_section = False
    computes_section = False
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        # Flags for sections
        if line.startswith("Operational Intensity Stats"):
            intensity_section = True
            computes_section = False
            summary_section = False
            computes_sub = None
            continue
        if line.startswith("Summary Stats"):
            intensity_section = False
            intensity_sub = None
            summary_section = True
            computes_section = False
            computes_sub = None
            continue
        if line.startswith("fJ/Compute"):
            intensity_section = False
            intensity_sub = None
            summary_section = False
            computes_section = True
            continue
        if not line:
            continue
        # Intensity section parsing
        if intensity_section:
            # print(line, intensity_sub, ":" in line)
            if ":" in line and not line.startswith("===") and intensity_sub is None:
                key, val = map(str.strip, line.split(":", 1))
                key = key.lower().replace(" ", "_")
                try:
                    summary[key] = float(val)
                except ValueError:
                    pass
            elif line.startswith("==="):
                intensity_sub = line.strip("= ")
                summary[f"{intensity_sub}_intensity"] = {}
            elif intensity_sub is not None and ":" in line:
                key, val = map(str.strip, line.split(":", 1))
                try:
                    summary[f"{intensity_sub}_intensity"][key] = float(val)
                except ValueError:
                    pass
            continue
        elif summary_section and (":" in line or "=" in line):
            try:
                key, val = map(str.strip, line.split(":", 1))
            except ValueError:
                key, val = map(str.strip, line.split("=", 1))
            if key == "Utilization":
                summary["Utilization (%)"] = float(val.strip().replace('%', ''))
            elif key == "Energy":
                parts = val.strip().split(" ")
                summary[f"Energy ({parts[1]})"] = float(parts[0])
            elif key == "Area":
                parts = val.strip().split(" ")
                summary[f"Area ({parts[1]})"] = float(parts[0])
            try:
                summary[key] = float(val)
            except ValueError:
                pass
        elif computes_section and "=" in line:
            key, val = map(str.strip, line.split("=", 1))
            try:
                summary[f"{key} fJ/compute"] = float(val)
            except ValueError:
                pass
    return summary


for accelerator, s in SIZES.items():
    loaded_config = [x for x in CONFIG]
    for i in range(len(loaded_config)):
        for k, v in s.items():
            if k in loaded_config[i]:
                loaded_config[i] = loaded_config[i].replace(k, str(v))
    os.system("rm -f ./example_designs/eyeriss_like/arch.yaml")
    with open("./example_designs/eyeriss_like/arch.yaml", "w") as f:
        f.writelines(loaded_config)

    for model, model_path in DATASETS.items():
        os.system(f"python3 run_example_designs.py --architecture eyeriss_like --problem {model_path} --n_jobs 12 --clear-outputs")
        os.system(f"python3 run_example_designs.py --architecture eyeriss_like --problem {model_path} --n_jobs 12")
        os.system(f"cp -r ./example_designs/eyeriss_like/outputs/ ./outputs/{accelerator}_{model}/")
    
        results = []
        for d in os.listdir("example_designs/eyeriss_like/outputs"):
            layer_name = f"{model}_{d}"
            layer_stats = parse_summary_stats(f"example_designs/eyeriss_like/outputs/{d}/timeloop-mapper.stats.txt")
            layer_stats["layer"] = layer_name
            results.append(layer_stats)
        
        df = pd.DataFrame(results).sort_values("layer")
        df.to_csv(f"./outputs/{accelerator}_{model}/layer_stats.csv", index=False)

    with open("./example_designs/eyeriss_like/arch.yaml", "w") as f:
        f.writelines(CONFIG)
print("Done")

    