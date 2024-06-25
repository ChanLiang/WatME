model=llama2-7b
synonym_method=llama2_13b_youdao_dict
synonym_clusters_path=./output/${synonym_method}/filtered_2d_list.json


# for gamma in 0.4 0.6
for gamma in 0.2 0.3 0.4 0.6
do
    # for delta in 2.0 3.0
    for delta in 2.0 3.0 4.0
    # for delta in 2 3 4 5 6
        do
        # for delta in 1
        # for delta in 2 
        # for delta in 3
        # for delta in 4
        # exp_name=./output/${synonym_method}/gsm8k_g${gamma}_d${delta}_result.jsonl
        exp_name=./output/gsm8k_${model}_g${gamma}_d${delta}_result.jsonl
        # exp_name=g${gamma}_d${delta}_T50
        echo $exp_name

        python3 -u parse_gsm8k_result.py \
        --log_path $exp_name 

        echo
    done
done