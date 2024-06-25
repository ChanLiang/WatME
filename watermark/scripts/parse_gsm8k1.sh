model=llama2-7b
# model=vicuna-7b-v1.5-16k

# synonym_method=llama2_13b_youdao_dict
# synonym_method=llama2_13b_supervised_llama2_chat_13b
# synonym_method=llama2_13b_supervised_llama2_70b
synonym_method=zeroshot-chatgpt

synonym_clusters_path=./output/${synonym_method}/filtered_2d_list.json


# for gamma in 0.4 0.6
# for gamma in 0.2 0.3 0.4 0.6
# for gamma in 0.3 0.4
for gamma in 0.3
do
for synonym_gamma in 0.3
# for synonym_gamma in 0.2 0.3
# for synonym_gamma in 0.3 0.4 0.5 0.6
do
    # for delta in 2.0 3.0
    for delta in 3.0
    # for delta in 2.0 3.0 4.0 5.0
    # for delta in 3.0 4.0 5.0
    # for delta in 2 3 4 5 6
    # for delta in 2.0
    # for delta in 5.0
    do
        # for delta in 1
        # for delta in 2 
        # for delta in 3
        # for delta in 4
        # exp_name=./output/${synonym_method}/gsm8k_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result_filter_answer_line_when_detecting.jsonl
        # # exp_name=./output/gsm8k_${model}_g${gamma}_d${delta}_removeRepetition_result_filter_answer_line_when_detecting.jsonl

        # this setting
        # exp_name=./output/${synonym_method}/gsm8k_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl
        # exp_name=./output/${synonym_method}/gsm8k_completely_${model}_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl

        # exp_name=./output/gsm8k_${model}_g${gamma}_d${delta}_removeRepetition_result.jsonl

        # exp_name=./output/${synonym_method}/gsm8k_sg${synonym_gamma}_g${gamma}_d${delta}_result_add_num.jsonl

        # exp_name=./output/${synonym_method}/gsm8k_sg${synonym_gamma}_g${gamma}_d${delta}_result.jsonl
        # exp_name=./output/${synonym_method}/gsm8k_g${gamma}_d${delta}_result.jsonl
        # exp_name=./output/gsm8k_${model}_g${gamma}_d${delta}_result.jsonl
        # exp_name=g${gamma}_d${delta}_T50

        # llama2-7b
        # watermark
        # exp_name=./output/gsm8k_${model}_g${gamma}_d${delta}_result.jsonl
        # exp_name=./output/gsm8k_${model}_g${gamma}_d${delta}_removeRepetition_result.jsonl

        # x-mark
        # exp_name=./output/${synonym_method}/gsm8k_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl
        # exp_name=./output/${synonym_method}/gsm8k_${model}_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl
        # exp_name=./output/${synonym_method}/gsm8k_completely_${model}_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl

        # vicuna-7b-v1.5-16k
        # exp_name=./output/gsm8k_${model}_g${gamma}_d${delta}_result.jsonl
        # exp_name=./output/${synonym_method}/gsm8k_${model}_sg${synonym_gamma}_g${gamma}_d${delta}_result.jsonl

        # exp_name=./output/gsm8k_${model}_g${gamma}_d${delta}_removeRepetition_result.jsonl
        # exp_name=./output/${synonym_method}/gsm8k_${model}_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl
        # exp_name=./output/${synonym_method}/gsm8k_completely_${model}_sg${synonym_gamma}_g${gamma}_d${delta}_removeRepetition_result.jsonl

        # stanford baseline
        # exp_name=./output/stanford_gsm8k_llama2-7b_g0.3_d3.0_removeRepetition_result.jsonl
        exp_name=./output/stanford_gsm8k_vicuna-7b-v1.5-16k_g0.3_d3.0_removeRepetition_result.jsonl

        echo $exp_name

        python3 -u parse_gsm8k_result.py \
        --log_path $exp_name 

        echo
    done
done
done