CKPT_PATH=/ckpts
cd /app/segment-caption-anything
python scripts/apps/sca_app.py \
+model=base_sca_multitask_v2 \
model.model_name_or_path=$CKPT_PATH \
model.lm_head_model_name_or_path=$(python scripts/tools/get_sub_model_name_from_ckpt.py $CKPT_PATH "lm")