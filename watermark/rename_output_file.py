import os

# directory = 'output'
# directory = 'output/llama2_13b_youdao_dict'

for directory in ['output/llama2_13b_youdao_dict', 'output/llama2_13b_supervised_llama2_70b', 'output/llama2_13b_supervised_llama2_chat_13b']:
    for filename in os.listdir(directory):
        # 如果文件名以 'gsm8k' 开头并且以 '_removeRepetition_result.jsonl' 结尾
        if filename.startswith('gsm8k') and filename.endswith('_removeRepetition_result.jsonl'):
            # 创建新文件名
            new_filename = filename.replace('_removeRepetition_result.jsonl', '_removeRepetition_result_filter_answer_line_when_detecting.jsonl')
            # 获取旧文件和新文件的完整路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file, new_file)

print("All file names have been changed successfully.")