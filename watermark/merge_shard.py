import os

shard_num = 8
# exp_name="./output/stanford_new_truthfulqa_llama2-7b_g0.3_d3.0_removeRepetition_result.jsonl"
# exp_name="./output/stanford_new_truthfulqa_vicuna-7b-v1.5-16k_g0.3_d3.0_removeRepetition_result.jsonl"

exp_name="./output/llama2_13b_youdao_dict/gsm8k_completely_llama2-7b_sg0.3_g0.3_d3.0_removeRepetition_result_samplingFalse"


lines = []
for shard_id in range(shard_num):
    if not os.path.exists(exp_name + f"_shard{shard_id}.jsonl"):
        print (f"{exp_name}{shard_id} not exists")
        continue
    cur = []
    with open(exp_name + f"_shard{shard_id}.jsonl", "r") as f:
        for line in f:
            lines.append(line.strip())
            cur.append(line.strip())
        print (shard_id, len(cur))


with open (exp_name + "_merged", "w") as f:
    for line in lines:
        f.write(line + "\n")

