#export DATA_DIR=/mnt/sda/allenlin/dataset/1mpx_rvt/
export DATA_DIR=/vol/NAS-Datasets/gen4/
export CKPT_PATH=./ckpts/rvt-s.ckpt
export MDL_CFG=small
export USE_TEST=0
export GPU_ID=0
export NUM_SEQUENCES=5 # note: n - 1

for i in $(seq 0 $NUM_SEQUENCES);
do
    mkdir -p output_images/sample_prediction_${i}/
    mkdir -p output_images/sample_prediction_tracking_${i}/
    mkdir -p output_images/sample_label_${i}/
    mkdir -p output_images/sample_${i}/

    python3 validation.py dataset=gen4 dataset.path=${DATA_DIR} \
        checkpoint=${CKPT_PATH} use_test_set=${USE_TEST} \
        wandb.project_name=RVT  wandb.group_name=1mpx \
        hardware.gpus=${GPU_ID} +experiment/gen4="${MDL_CFG}.yaml" \
        model.postprocess.confidence_threshold=0.001 \
        batch_size.eval=1 hardware.num_workers.eval=1 custom.select_sequence=${i}

    #continue;

    echo "converting ${i} sequence"
    ffmpeg -framerate 10 -pattern_type glob -i "output_images/sample_prediction_tracking_${i}/*.jpg" -c:v libx264 -pix_fmt yuv420p output_images/sample_prediction_tracking_${i}.mp4
    ffmpeg -framerate 10 -pattern_type glob -i "output_images/sample_prediction_${i}/*.jpg" -c:v libx264 -pix_fmt yuv420p output_images/sample_prediction_${i}.mp4
    ffmpeg -framerate 10 -pattern_type glob -i "output_images/sample_label_${i}/*.jpg" -c:v libx264 -pix_fmt yuv420p output_images/sample_label_${i}.mp4
    ffmpeg -framerate 10 -pattern_type glob -i "output_images/sample_${i}/*.jpg" -c:v libx264 -pix_fmt yuv420p output_images/sample_${i}.mp4
done
