import os
import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt

architecture = "eyeriss_like"

os.system(f"rm -rf ./{architecture}_outputs; mkdir ./{architecture}_outputs")

with open(f"./example_designs/{architecture}/arch_base.yaml", "r") as f:
    CONFIG = f.readlines()

DATASETS = {
    "ConvNeXt": "ConvNeXt",
    "AlexNet": "CONV/AlexNet",
    "Resnet18": "resnet18",
    "ViT": "vision_transformer",
}

SIZES = {
    "V100TensorCore": {"__meshX__": 8, "__meshY__": 16, "__width__": 64, "__datawidth__": 64},
    "A100TensorCore": {"__meshX__": 16, "__meshY__": 32, "__width__": 160, "__datawidth__": 32},
    "3060TensorCore": {"__meshX__": 16, "__meshY__": 32, "__width__": 24, "__datawidth__": 8},
    "3090TensorCore": {"__meshX__": 16, "__meshY__": 32, "__width__": 48, "__datawidth__": 8},
    "14x12": {"__meshX__": 14, "__meshY__": 12, "__width__": 64, "__datawidth__": 16},
    "8x21": {"__meshX__": 8, "__meshY__": 21, "__width__": 64, "__datawidth__": 16},
    "21x8": {"__meshX__": 21, "__meshY__": 8, "__width__": 64, "__datawidth__": 16},
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
    os.system(f"rm -f ./example_designs/{architecture}/arch.yaml")
    with open(f"./example_designs/{architecture}/arch.yaml", "w") as f:
        f.writelines(loaded_config)

    for model, model_path in DATASETS.items():
        os.system(f"python3 run_example_designs.py --architecture {architecture} --problem {model_path} --n_jobs 16 --clear-outputs")
        os.system(f"python3 run_example_designs.py --architecture {architecture} --problem {model_path} --n_jobs 16")
        os.system(f"cp -r ./example_designs/{architecture}/outputs/ ./{architecture}_outputs/{accelerator}_{model}/")
    
        results = []
        for d in os.listdir(f"example_designs/{architecture}/outputs"):
            layer_name = f"{model}_{d}"
            layer_stats = parse_summary_stats(f"example_designs/{architecture}/outputs/{d}/timeloop-mapper.stats.txt")
            layer_stats["layer"] = layer_name
            results.append(layer_stats)
        
        df = pd.DataFrame(results).sort_values("layer")
        df.to_csv(f"./{architecture}_outputs/{accelerator}_{model}/layer_stats.csv", index=False)

    with open(f"./example_designs/{architecture}/arch.yaml", "w") as f:
        f.writelines(CONFIG)
print("Done")

    