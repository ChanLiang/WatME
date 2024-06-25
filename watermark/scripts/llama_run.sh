export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

# export HF_HOME=/apdcephfs_cq3/share_2934111/chenliang/cache
# export HF_DATASETS_CACHE=/apdcephfs_cq3/share_2934111/chenliang/hf/datasets_cache

model_name_or_path=/apdcephfs_cq3/share_2934111/chenliang/cache/llama/30B
oracle_model_name=/apdcephfs_cq3/share_2934111/chenliang/cache/llama/65B

num_samples=200

# for debug
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0
# python3 -u watermark_exp.py \
# --model_name_or_path $model_name_or_path \
# --oracle_model_name $oracle_model_name \
# --load_fp16 True \
# --use_synonyms False \
# --num_samples $num_samples \
# --gamma 0.4 \
# --delta 4 \
# --max_new_tokens 50 \
# --debug_mode True \
# --exp_name llama_debug
# --exp_name llama_debug 1>log/llama_debug.log 2>&1


# for gamma in 0.4 0.6
for gamma in 0.4
do
    for delta in 2 4
    do
        # for T in 200 
        for T in 100 
        do
            python3 -u watermark_exp.py \
            --model_name_or_path $model_name_or_path \
            --oracle_model_name $oracle_model_name \
            --load_fp16 True \
            --use_synonyms False \
            --num_samples $num_samples \
            --gamma $gamma \
            --delta $delta \
            --max_new_tokens $T \
            --exp_name llama30B_g${gamma}_d${delta}_bl_sample${num_samples}_T${T}_fast 1>log/llama30B_g${gamma}_d${delta}_bl_sample${num_samples}_T${T}_fast.log 2>&1
            wait
        done
    done
done


model_name_or_path=/apdcephfs_cq3/share_2934111/chenliang/cache/llama/65B

for gamma in 0.4
do
    for delta in 2 4
    do
        for T in 100 
        do
            python3 -u watermark_exp.py \
            --model_name_or_path $model_name_or_path \
            --oracle_model_name $oracle_model_name \
            --load_fp16 True \
            --use_synonyms False \
            --num_samples $num_samples \
            --gamma $gamma \
            --delta $delta \
            --max_new_tokens $T \
            --exp_name llama65B_g${gamma}_d${delta}_bl_sample${num_samples}_T${T}_fast 1>log/llama65B_g${gamma}_d${delta}_bl_sample${num_samples}_T${T}_fast.log 2>&1
            wait
        done
    done
done