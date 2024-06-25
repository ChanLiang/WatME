import json

for d in ['2.0', '3.0', '4.0', '5.0']:
    exp_name = f"output/llama2_13b_youdao_dict/gsm8k_sg0.3_g0.3_d{d}_result.jsonl"
    with open(exp_name, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1319, f"{exp_name} has {len(lines)} lines, not 1319"

    w1 = open(exp_name[:-6] + '_wo_watermark_output', "w") 
    w2 = open(exp_name[:-6] + '_w_watermark_output', "w")

    for line in lines: # 1319, 1 hour...
        result = json.loads(line)
        wo_watermark_output = result["wo_watermark_output"].replace('\n', ';').strip()
        w_watermark_output = result["w_watermark_output"].replace('\n', ';').strip()
    
        # print (wo_watermark_output)
        # print (w_watermark_output)
        w1.write(wo_watermark_output + '\n')
        w2.write(w_watermark_output + '\n')
            