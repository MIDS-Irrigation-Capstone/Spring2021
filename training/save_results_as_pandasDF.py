import json

from matplotlib import pyplot as plt
import pandas as pd
import pickle
from s3fs.core import S3FileSystem

MODEL_DIR = "mids-capstone-irrigation-detection/models"

s3_file = S3FileSystem()

# Models are model_name: description
models = {
    "supervised_baseline": "Balanced Dataset",
    "supervised_baseline_ex": "Balanced Extended Labels",
    "supervised_baseline_pretrained": "ImageNet Pretraining",
    "supervised_baseline_pretrained_ex": "ImageNet Pretraining with Extended Labels"
}

def f_scores(scores):
    precision = scores[6]
    recall = scores[7]
    if precision + recall == 0.0:
        return [0.0, 0.0]
    f1 = (2 * precision * recall) / (precision + recall)
    beta = 0.5
    f05 = ((1 + beta) * precision * recall) / (beta * precision + recall)
    return [f1, f05]


def load_results():
    data = []
    for file in s3_file.listdir(MODEL_DIR):
        if file['type'] != 'directory':
            continue
        model_type = file["name"].split("/")[-1]
        for result in s3_file.glob(file["name"]+"/*.json"):
            model_type = result.split("/")[-2]
            percent = result.split("/")[-1].split("_")[0]
            day = result.split("/")[-1].split("-")[-1].split(".")[0]
            r = json.load(s3_file.open(result))
            score = [model_type, int(percent), r.get("pretrain", False), r["architecture"]] + r["score"] + f_scores(r["score"])
            data.append(score)
    df = pd.DataFrame(data)
    df.columns = ["model_type", "split_percent","pretrain", "architecture","loss","tp","fp","tn","fn","accuracy","precision","recall","auc","f1","f0.5"]
    df.split_percent = pd.to_numeric(df.split_percent, errors='coerce')
    return df

def create_results_frame():
    df = load_results()
    df.to_csv('results.csv')

if __name__ == "__main__":
    create_results_frame()
