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

    assert len(lines) == 817, (len(lines))

    detection_list = []
    w_lens, wo_lens, w_green_ratios, wo_green_ratios = [], [], [], []
    for line in lines:
        result = json.loads(line)
        if "w_watermark_detection" not in result:
            continue
        i = 0
        assert result["w_watermark_detection"][i][0] == 'z-score', (result["w_watermark_detection"][i][0])
        # w_watermark_detection = result["w_watermark_detection"][i][-1]
        # wo_watermark_detection = result["wo_watermark_detection"][i][-1]
        # detection_list.append([w_watermark_detection, wo_watermark_detection])
        # w_lens.append(float(result["w_watermark_detection"][0][-1]))
        # wo_lens.append(float(result["wo_watermark_detection"][0][-1]))
        # w_green_ratios.append(float(result["w_watermark_detection"][2][-1].strip('%')) / 100)
        # wo_green_ratios.append(float(result["wo_watermark_detection"][2][-1].strip('%')) / 100)

        w_watermark_detection = result["w_watermark_detection"][i][-1]
        wo_watermark_detection = result["wo_watermark_detection"][i][-1]
        print (w_watermark_detection, wo_watermark_detection)
        detection_list.append([w_watermark_detection, wo_watermark_detection])
        w_lens.append(float(result["w_watermark_detection"][0][-1]))
        wo_lens.append(float(result["wo_watermark_detection"][0][-1]))
        w_green_ratios.append(float(result["w_watermark_detection"][2][-1].strip('%')) / 100)
        wo_green_ratios.append(float(result["wo_watermark_detection"][2][-1].strip('%')) / 100)
    # w_list = [x[0] for x in detection_list]
    # wo_list = [x[1] for x in detection_list]
    w_list = [1 - float(x[0]) for x in detection_list]
    wo_list = [1 - float(x[1]) for x in detection_list]
    print ('total: {}'.format(len(detection_list)))
    print ('avg watermarked z-score: {}'.format(sum([float(x[0]) for x in detection_list])/len(detection_list)))
    print ('avg non-watermarked z-score: {}'.format(sum([float(x[1]) for x in detection_list])/len(detection_list)))
    best_threshold, best_accuracy, best_precision, best_recall, best_f1, best_auroc = find_best_threshold(w_list, wo_list)
    print("best_threshold: {}, auroc: {}, acc: {}, P/R/F1: {}/{}/{}".format(best_threshold, best_accuracy, best_auroc, best_precision, best_recall, best_f1))
    print ('avg w length: {}'.format(sum(w_lens)/len(w_lens)))
    print ('avg wo length: {}'.format(sum(wo_lens)/len(wo_lens)))
    print ('avg w green ratio: {}'.format(sum(w_green_ratios)/len(w_green_ratios)))
    print ('avg wo green ratio: {}'.format(sum(wo_green_ratios)/len(wo_green_ratios)))