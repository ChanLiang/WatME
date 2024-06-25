model=llama2-7b
# model=llama2_13b
# model=vicuna-7b-v1.5-16k

# synonym_method=llama2_13b_supervised_llama2_chat_13b
# synonym_method=llama2_13b_supervised_llama2_70b
synonym_method=llama2_13b_youdao_dict
# synonym_method=zeroshot-chatgpt
synonym_clusters_path=./output/${synonym_method}/filtered_2d_list.json

num_shard=8

# for gamma in 0.4 0.6
# for gamma in 0.2 0.3 0.4 0.6
# for gamma in 0.3 0.4
for gamma in 0.3
# for gamma in 0.4
do
for synonym_gamma in 0.3
# for synonym_gamma in 0.2 0.3
# for synonym_gamma in 0.3 0.4 0.5 0.6
do
    # for delta in 2.0 3.0
    # for delta in 2.0 3.0 4.0 5.0
    # for delta in 3.0 4.0 5.0
    # for delta in 3.0 5.0
    # for delta in 2 3 4 5 6
    # for delta in 2.0
    for delta in 3.0
    # for delta in 5.0
        do
        # for delta in 1
        # for delta in 2 
        # for delta in 3
        # for delta in 4
        # exp_name=./output/truthfulqa_${model}_g${gamma}_d${delta}_removeRepetition_result.jsonl
        # exp_name=./output/${synonym_method}/truthfulqa_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl

        # vicuna
        # exp_name=./output/truthfulqa_llama2_13b_g${gamma}_d${delta}_removeRepetition_result.jsonl
        # exp_name=./output/${synonym_method}/truthfulqa_vicuna-7b-v1.5-16k_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl
        # exp_name=./output/${synonym_method}/truthfulqa_completely_vicuna-7b-v1.5-16k_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl
        
        # exp_name=./output/${synonym_method}/truthfulqa_sg${synonym_gamma}_g${gamma}_d${delta}_result.jsonl
        # exp_name=./output/${synonym_method}/truthfulqa_g${gamma}_d${delta}_result.jsonl
        # exp_name=./output/truthfulqa_${model}_g${gamma}_d${delta}_result.jsonl
        # exp_name=g${gamma}_d${delta}_T50

        # exp_name=./output/stanford_truthfulqa_llama2-7b_g${gamma}_d${delta}_removeRepetition_result.jsonl
        # exp_name=./output/stanford_truthfulqa_vicuna-7b-v1.5-16k_g${gamma}_d${delta}_removeRepetition_result.jsonl

        # dist
        # exp_name=./output/stanford_new_truthfulqa_llama2-7b_g${gamma}_d${delta}_removeRepetition_result.jsonl_shard
        # exp_name=./output/stanford_new_truthfulqa_vicuna-7b-v1.5-16k_g${gamma}_d${delta}_removeRepetition_result.jsonl_shard

        # 3.28
        # exp_name=output/truthfulqa_328_llama2-7b_g0.3_d4.0_removeRepetition_result.jsonl
        exp_name=output/llama2_13b_youdao_dict/truthfulqa_328_completely_llama2-7b_sg0.3_g0.3_d4.0_removeRepetition_result.jsonl
        num_shard=0

        echo $exp_name

        python3 -u parse_truthfulqa_result_dist.py \
        --log_path $exp_name \
        --shard_num $num_shard 

        echo
    done
done
done