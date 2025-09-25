# read debug flag
DEBUG=0
while getopts "d" flag; do
        case "${flag}" in
                d) DEBUG=1;;
        esac
done
shift $((OPTIND - 1))

# scalinglaw_data_dir=$WORK/data/improved/scalinglaw

# # split_0-163_ic128
# # split_163-327_ic64
# # split_327-655_ic32
# # split_655-1311_ic16
# # split_1311-2622_ic8
# # split_2622-5244_ic4
# # split_5244-10489_ic2

# train_data_dirs=(
#     $scalinglaw_data_dir/split_163-327_ic64/train
# )

train_data_dirs=(
#     ./data/improved/skew_mixedp_ic16/train
#     ./data/improved/final_skew40/train
#     ./data/improved/final_skew40/train_z5_z10
#     ./data/improved/final_base40/train
#     ./data/improved/final_base40/train_z5_z10
      ./data/huggingface/train
)

# --- START: Modified section (replaces jq) ---
# Goal: Convert the bash array to a JSON array string.
# Example: (path1 path2) -> ["path1","path2"]

train_data_dirs_json="[" # Start the JSON array string
num_dirs=${#train_data_dirs[@]} # Get the total number of directories
i=0
for dir in "${train_data_dirs[@]}"; do
    train_data_dirs_json+="\"$dir\"" # Add the directory path in quotes
    i=$((i + 1))
    if [ "$i" -lt "$num_dirs" ]; then
        train_data_dirs_json+="," # Add a comma if it's not the last element
    fi
done
train_data_dirs_json+="]" # Close the JSON array string
# --- END: Modified section ---


ulimit -n 99999
if [ "$DEBUG" -eq 0 ]; then

        TOTAL_CORES=$(nproc)
        CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
        CORES_PER_JOB=$(( $CORES_PER_GROUP / 3 ))

        # CUDA_DEVICES=0,1,2,3
        CUDA_DEVICES=3,5,6,7
        NUM_DEVICES=$(tr ',' '\n' <<< "$CUDA_DEVICES" | wc -l)

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$CORES_PER_JOB torchrun \
                --nproc-per-node $NUM_DEVICES \
                --master-port 29501 \
                scripts/patchtst/train.py \
                train_data_dirs=$train_data_dirs_json \
                patchtst.mode=pretrain \
                patchtst.context_length=512 \
                patchtst.patch_length=8 \
                patchtst.patch_stride=8 \
                patchtst.d_model=48 \
                patchtst.norm_type=rmsnorm \
                patchtst.channel_attention=true \
                patchtst.use_dynamics_embedding=false \
                patchtst.channel_consistent_masking=false \
                patchtst.max_wavelength=500 \
                patchtst.rope_percent=0.75 \
                patchtst.loss=mse \
                train.per_device_train_batch_size=2048 \
                train.max_steps=200_000 \
                train.save_steps=20_000 \
                train.log_steps=1_000 \
                train.warmup_ratio=0.1 \
                train.torch_compile=true \
                scheduler.enabled=false \
                train.output_dir=./checkpoints \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/patchtst/train.py \
                run_name=DEBUG \
                shuffle_buffer_length=100 \
                patchtst.mode=pretrain \
                train.ddp_backend=null \
                train.torch_compile=false \
                "$@"
fi
