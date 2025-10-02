# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/compress.py \
#     --checkpoint_dir checkpoints \
#     --ar_model_dir Cosmos-Predict1-12B \
#     --input_type video \
#     --input_image_or_video_path assets/autoregressive/input.mp4 \
#     --temperature 1.0 \
#     --offload_diffusion_decoder \
#     --offload_tokenizer \
#     --offload_ar_model \
#     --disable_guardrail \
#     --video_save_folder outputs/compress-12B_diffusion_decoder-9frames \
#     --video_save_name autoregressive-12b-base \
#     --buffer_frames 9

# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/compress.py \
#     --checkpoint_dir checkpoints \
#     --ar_model_dir Cosmos-Predict1-12B \
#     --input_type video \
#     --input_image_or_video_path assets/autoregressive/input.mp4 \
#     --temperature 1.0 \
#     --offload_diffusion_decoder \
#     --offload_tokenizer \
#     --offload_ar_model \
#     --disable_guardrail \
#     --video_save_folder outputs/compress-12B_diffusion_decoder-17frames \
#     --video_save_name autoregressive-12b-base \
#     --buffer_frames 17

# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/compress.py \
#     --checkpoint_dir checkpoints \
#     --ar_model_dir Cosmos-Predict1-12B \
#     --input_type video \
#     --input_image_or_video_path assets/autoregressive/input.mp4 \
#     --temperature 1.0 \
#     --offload_diffusion_decoder \
#     --offload_tokenizer \
#     --offload_ar_model \
#     --disable_guardrail \
#     --video_save_folder outputs/compress-12B_diffusion_decoder-25frames \
#     --video_save_name autoregressive-12b-base \
#     --buffer_frames 25

# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/compress.py \
#     --checkpoint_dir checkpoints \
#     --ar_model_dir Cosmos-Predict1-12B \
#     --input_type video \
#     --input_image_or_video_path datasets/BasketballDrive_BT601_frist33_1280x704.mp4 \
#     --temperature 1.0 \
#     --offload_diffusion_decoder \
#     --offload_tokenizer \
#     --offload_ar_model \
#     --disable_guardrail \
#     --video_save_folder outputs/compress-12B_diffusion_decoder-basketball-25frames \
#     --video_save_name compress-BasketballDrive_BT601_frist33_1280x704 \
#     --buffer_frames 25


video_path=datasets/BasketballDrive_1024x640_50.yuv
model=Cosmos-Predict1-12B
video_save_name=compress-BasketballDrive_BT601_first33_1024x640_yuv

save_folder=outputs/tokenizer_8x16x16_720p
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
    --disable_entropy_coding \
    --video_save_folder $save_folder \
    --video_save_name $video_save_name \
    --buffer_frames 1 > $save_folder/log.txt 

save_folder=outputs/tokenizer_diffusion_decoder_8x16x16_720p
mkdir $save_folder
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
    --disable_entropy_coding \
    --video_save_folder $save_folder \
    --video_save_name $video_save_name \
    --buffer_frames 1  > $save_folder/log.txt 

