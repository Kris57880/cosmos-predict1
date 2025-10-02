export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
width=1024
height=640
video_path=datasets/BasketballDrive_${width}x${height}_50.yuv
model=Cosmos-Predict1-4B
video_save_name=compress-BasketballDrive_BT601_first33_${width}x${height}_yuv

save_folder=outputs/compress-4B-basketball-25frames
mkdir -p $save_folder

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/compress.py \
    --checkpoint_dir checkpoints \
    --ar_model_dir $model \
    --input_type video \
    --input_image_or_video_path $video_path \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --offload_ar_model \
    --disable_guardrail \
    --disable_diffusion_decoder \
    --video_save_folder $save_folder \
    --video_save_name $video_save_name \
    --buffer_frames 25 > $save_folder/log.txt
