export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

# export HF_HOME=/apdcephfs_cq3/share_2934111/chenliang/cache
# export HF_DATASETS_CACHE=/apdcephfs_cq3/share_2934111/chenliang/hf/datasets_cache

model_name_or_path=llama2-7b
# model_name_or_path=vicuna-7b-v1.5-16k
paraphraser=llama2-chat-13b

# synonym_method=llama2_13b_supervised_llama2_70b
# synonym_method=zeroshot-chatgpt

synonym_method=llama2_13b_supervised_llama2_chat_13b
# synonym_method=llama2_13b_youdao_dict

# synonym_clusters_path=./output/${synonym_method}/filtered_2d_list.json
synonym_clusters_path=./output/${synonym_method}/completely_filtered_2d_list1.json

# num_samples=10000
num_samples=200
few_shot=True

# debug_mode=True
debug_mode=False

# use_synonyms=True
use_synonyms=False

# max_new_tokens=150
max_new_tokens=128

remove_output_repetition=True
# remove_output_repetition=False

# filter_answer_line_when_detecting=True
filter_answer_line_when_detecting=False

sampling_temp=0.0
# sampling_temp=1.0

# use_sampling=True
use_sampling=False

fix_seed=True

# for gamma in 0.2
for gamma in 0.3
# for gamma in 0.4
# for gamma in 0.6
do
# for synonym_gamma in 0.2 0.3 0.4 0.5 0.6
# for synonym_gamma in 0.2 0.3
# for synonym_gamma in 0.4 0.5 0.6
# for synonym_gamma in 0.5 0.6 0.7
for synonym_gamma in 0.3
# for synonym_gamma in 0.4
# for synonym_gamma in 0.4 0.5 0.6 0.7
do
    # for delta in 3.0 4.0 5.0
    # for delta in 2.0 3.0 4.0 5.0
    # for delta in 3.0 4.0 5.0 6.0
    for delta in 3.0
    # for delta in 4.0
    # for delta in 5.0
    do
    # exp_name=gsm8k_${synonym_method}_${use_synonyms}_sg${synonym_gamma}_g${gamma}_d${delta}
    # exp_name=gsm8k_${synonym_method}_${use_synonyms}_sg${synonym_gamma}_g${gamma}_d${delta}_T50_add_num
    # exp_name=gsm8k_${synonym_method}_${use_synonyms}_sg${synonym_gamma}_g${gamma}_d${delta}_remove_repetition${remove_output_repetition}

    # exp_name=gsm8k_completely_${model_name_or_path}_${synonym_method}_${use_synonyms}_sg${synonym_gamma}_g${gamma}_d${delta}_remove_repetition${remove_output_repetition}_filter_answer_line_when_detecting${filter_answer_line_when_detecting}_sampling${use_sampling}
    # exp_name=1.25_fix_seed${fix_seed}_gsm8k_completely_${model_name_or_path}_${synonym_method}_${use_synonyms}_sg${synonym_gamma}_g${gamma}_d${delta}_remove_repetition${remove_output_repetition}_filter_answer_line_when_detecting${filter_answer_line_when_detecting}_sampling${use_sampling}
    # exp_name=3.27_para_fixseed${fix_seed}_gsm8k_complete_${model_name_or_path}_${synonym_method}_${use_synonyms}_sg${synonym_gamma}_g${gamma}_d${delta}_remove_repetition${remove_output_repetition}_filter_answer_line_when_detecting${filter_answer_line_when_detecting}_sampling${use_sampling}
    exp_name=3.28_para_fixseed${fix_seed}_gsm8k_complete_${model_name_or_path}_${synonym_method}_${use_synonyms}_sg${synonym_gamma}_g${gamma}_d${delta}_remove_repetition${remove_output_repetition}_filter_answer_line_when_detecting${filter_answer_line_when_detecting}_sampling${use_sampling}
    # exp_name=g${gamma}_d${delta}_T50
    echo $exp_name

    export CUDA_VISIBLE_DEVICES=0,1
    # export CUDA_VISIBLE_DEVICES=2,3
    # export CUDA_VISIBLE_DEVICES=4,5
    # export CUDA_VISIBLE_DEVICES=6,7
    
    python3 -u detect_watermark_exp_gsm8k_para1.py \
    --remove_output_repetition $remove_output_repetition \
    --filter_answer_line_when_detecting $filter_answer_line_when_detecting \
    --math True \
    --fix_seed $fix_seed \
    --exp_name $exp_name \
    --model_name_or_path $model_name_or_path \
    --paraphraser $paraphraser \
    --use_synonyms $use_synonyms \
    --max_new_tokens $max_new_tokens \
    --synonym_clusters_path $synonym_clusters_path \
    --few_shot $few_shot \
    --synonym_method $synonym_method \
    --use_sampling False \
    --use_sampling $use_sampling \
    --sampling_temp $sampling_temp \
    --n_beams 1 \
    --load_fp16 True \
    --num_samples $num_samples \
    --gamma $gamma \
    --synonym_gamma $synonym_gamma \
    --delta $delta \
    --debug_mode $debug_mode 1>log/${exp_name}.log 2>&1

    # --debug_mode $debug_mode

    done
done
done
