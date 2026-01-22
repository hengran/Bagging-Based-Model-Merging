

export WANDB_MODE=disabled
train_data="\
    data/jsonl_output/retrieval \
    data/jsonl_output/sts/sts.jsonl \
    data/jsonl_output/classification-no_in_batch_neg \
    data/jsonl_output/clustering-no_in_batch_neg "

num_train_epochs=1
per_device_train_batch_size=32

num_gpus=8

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path Qwen/Qwen3-0.6B \
    --cache_dir $HF_HUB_CACHE \
    --use_lora False \
    --lora_rank 32 \
    --lora_alpha 32 \
    --use_mrl False \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --save_merged_lora_model True \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Given a query, retrieve passages that are relevant to the query.' \
    --query_instruction_format 'Instruct: {}\nQuery: {}' \
    --knowledge_distillation False \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
"

training_args="\
    --output_dir ./Qwen3-0.6B_original_0.6B_without_mr_plus_code \
    --overwrite_output_dir \
    --learning_rate 5e-5 \
    --fp16 \
    --mrl_dims 256 512 768 1024 2048 2560 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed examples/finetune/ds_stage1.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --gradient_accumulation_steps 1 \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
"
cmd="deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60001 --module \
    FlagEmbedding.finetune.embedder.decoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
