# debug=True
debug=False

# only works in the synonyms inference stage
# model=llama2-chat-70b
# model=llama2-70b
model=llama2-13b

bz=20
# bz=32

method=lm
# method=youdao


# for exp_name in llama2_13b_supervised_llama2_70b llama2_13b_supervised_llama2_chat_13b llama2_13b_youdao_dict
for exp_name in zeroshot-chatgpt
do
echo
echo $exp_name
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0,1,2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4
# export CUDA_VISIBLE_DEVICES=6,7
python3 -u build_cluster_llama.py \
--exp_name $exp_name \
--synonym_model $model \
--debug $debug \
--method $method \
--bz $bz 
done