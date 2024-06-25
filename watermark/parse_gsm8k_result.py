import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")
    parser.add_argument("--log_path", type=str)
    args = parser.parse_args()
    return args

def find_best_threshold(positive_predictions, negative_predictions):
    # print (type(positive_predictions), type(negative_predictions)) # <class 'list'> <class 'list'>
    # print (positive_predictions[:10], negative_predictions[:10])
    positive_predictions = [float(x) for x in positive_predictions]
    negative_predictions = [float(x) for x in negative_predictions]
    # 将正类和负类的预测值合并，并按照从小到大排序
    all_predictions = positive_predictions + negative_predictions
    all_predictions.sort()

    best_threshold = None
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    auroc = roc_auc_score([1]*len(positive_predictions) + [0]*len(negative_predictions), positive_predictions + negative_predictions)

    # 逐个尝试每个阈值，计算准确率、精确率、召回率和 F1 值
    for threshold in all_predictions:
        positive_results = [1 if p >= threshold else 0 for p in positive_predictions]
        negative_results = [1 if n >= threshold else 0 for n in negative_predictions]

        true_positives = sum(positive_results)
        true_negatives = len(negative_results) - sum(negative_results)
        total_samples = len(positive_results) + len(negative_results)
        accuracy = (true_positives + true_negatives) / total_samples

        precision = true_positives / (true_positives + sum(negative_results))
        recall = true_positives / len(positive_results)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        y_true = np.concatenate((np.ones(len(positive_results)), np.zeros(len(negative_results))))
        y_scores = np.concatenate((positive_predictions, negative_predictions))

        # if accuracy > best_accuracy:
        if f1 > best_f1:
        # if auroc > best_auroc:
            best_threshold = threshold
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    return best_threshold, best_accuracy, best_precision, best_recall, best_f1, auroc

if __name__ == "__main__":
    args = parse_args()
    with open(args.log_path, "r") as f:
        lines = f.readlines()

    print ('len(lines) = ', len(lines))
    if len(lines) > 1319:
        lines = lines[:1319]
    assert len(lines) == 1319, (len(lines))

    acc_list, detection_list = [], []
    lens, green_ratios = [], []
    for line in lines:
        result = json.loads(line)
        w_watermark_acc = result["w_watermark_acc"]
        wo_watermark_acc = result["wo_watermark_acc"]
        # assert result["w_watermark_detection"][3][0] == 'z-score', (result["w_watermark_detection"][3][0])
        # i = 3
        i = 0
        if result["w_watermark_detection"][i][0] != 'z-score':
            print (result["w_watermark_detection"])
            continue
        w_watermark_detection = result["w_watermark_detection"][i][-1]
        wo_watermark_detection = result["wo_watermark_detection"][i][-1]
        acc_list.append([w_watermark_acc, wo_watermark_acc])
        detection_list.append([w_watermark_detection, wo_watermark_detection])
        lens.append(float(result["w_watermark_detection"][0][-1]))
        green_ratios.append(float(result["w_watermark_detection"][2][-1].strip('%')) / 100)
    w_list = [1 - float(x[0]) for x in detection_list]
    wo_list = [1 - float(x[1]) for x in detection_list]
    print ('avg watermarked z-score: {}'.format(sum([float(x[0]) for x in detection_list])/len(detection_list)))
    print ('avg non-watermarked z-score: {}'.format(sum([float(x[1]) for x in detection_list])/len(detection_list)))
    wo_acc = sum([float(x[1]) for x in acc_list])/len(acc_list)
    w_acc = sum([float(x[0]) for x in acc_list])/len(acc_list)
    print("w_acc: {}, wo_acc: {}".format(w_acc, wo_acc))
    best_threshold, best_accuracy, best_precision, best_recall, best_f1, best_auroc = find_best_threshold(w_list, wo_list)
    print("best_threshold: {}, auroc: {}, acc: {}, P/R/F1: {}/{}/{}".format(best_threshold, best_accuracy, best_auroc, best_precision, best_recall, best_f1))
    print ('avg length: {}'.format(sum(lens)/len(lens)))
    print ('avg green ratio: {}'.format(sum(green_ratios)/len(green_ratios)))