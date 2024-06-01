import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys


with open("result.json", "r", encoding="utf-8") as f:
    result = json.load(f)
orgScore = result[-1]
keys = list(orgScore.keys())
exps = {}
exp_folders = ["qwen1.5-7b-lora-qk", "qwen1.5-7b-lora-all", "qwen1.5-7b-prompt"]
for e in exp_folders:
    results = {}
    checkpoints = [folder for folder in os.listdir(e) if folder.startswith("checkpoint")]
    results["0"] = orgScore
    for c in checkpoints:
        step = c.split("-")[-1]
        with open(os.path.join(e, c, "result.json"), "r", encoding="utf-8") as f:
            result = json.load(f)
        score = result[-1]
        results[step] = score
    exps[e] = results
print(exps)
for imageTitle in keys:
    lines = []
    for expName, results in exps.items():
        xs = []
        ys = []
        for step, allScore in results.items():
            score = allScore[imageTitle]
            xs.append(int(step))
            ys.append(score)
        lines.append((xs, ys, expName))
    plt.figure()
    for xs, ys, label in lines:
        plt.plot(xs, ys, label=label)
        for x, y in zip(xs, ys):
            plt.text(x, y, f"{y:.2f}")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("score")
    plt.title(imageTitle)
    plt.savefig(f"{imageTitle}.png")

