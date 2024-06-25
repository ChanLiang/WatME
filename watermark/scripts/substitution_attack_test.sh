#!/bin/bash

PYTHON_INTERPRETER=python3
SCRIPT_PATH="substitution_attack_test.py"

MODEL_NAME_OR_PATH=llama2-7b
# MODEL_NAME_OR_PATH=vicuna-7b-v1.5-16k

# ANSWER_PATH=output/gsm8k_llama2-7b_g0.3_d3.0_removeRepetition_result.jsonl
# ANSWER_PATH=output/llama2_13b_youdao_dict/gsm8k_completely_sg0.3_g0.3_d3.0_result.jsonl
ANSWER_PATH=./output/llama2_13b_youdao_dict/gsm8k_completely_llama2-7b_sg0.3_g0.3_d3.0_removeRepetition_result_samplingFalse_merged


# ANSWER_PATH=output/gsm8k_vicuna-7b-v1.5-16k_g0.3_d3.0_removeRepetition_result.jsonl

# ANSWER_PATH=output/gsm8k_llama2-7b_g0.3_d3.0_result.jsonl
# ANSWER_PATH=output/llama2_13b_youdao_dict/gsm8k_completely_sg0.3_g0.3_d3.0_result.jsonl

# USE_SYNONYMS=False
USE_SYNONYMS=True

SYNONYM_GAMMA=0.3
GAMMA=0.3
DELTA=3

replace_ratio=0.0
# replace_ratio=0.1
# replace_ratio=0.2
# replace_ratio=0.3
# replace_ratio=0.4
# replace_ratio=0.5

# use_synonyms=True

synonym_method=llama2_13b_youdao_dict
synonym_method=llama2_13b_supervised_llama2_chat_13b
SYNONYM_CLUSTERS_PATH=./output/${synonym_method}/completely_filtered_2d_list1.json


$PYTHON_INTERPRETER $SCRIPT_PATH --model_name_or_path $MODEL_NAME_OR_PATH \
                                 --answer_path $ANSWER_PATH \
                                 --use_synonyms $USE_SYNONYMS \
                                 --synonym_gamma $SYNONYM_GAMMA \
                                 --gamma $GAMMA \
                                 --delta $DELTA \
                                 --replace_ratio $replace_ratio \
                                 --synonym_clusters_path $SYNONYM_CLUSTERS_PATH \