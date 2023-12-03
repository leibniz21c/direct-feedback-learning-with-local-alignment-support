
BATCH_SIZE=64
ARCHITECTURE=resnet18
OPTIMIZER=Adam

DATASET=TINYIMAGENET
NUM_EPOCHS=500

for LEARNING in dfa las bp ltas
do
    for LEARNING_RATE in 1e-4 5e-5 1e-5
    do
        for WEIGHT_DECAY in 1e-3 1e-4 1e-5
        do
            RESULT_PATH=saved/${ARCHITECTURE}/${DATASET}/${OPTIMIZER}/${LEARNING}/lr=${LEARNING_RATE}-wd=${WEIGHT_DECAY}
            python3 train.py \
                --result_path ${RESULT_PATH} \
                --dataset ${DATASET} \
                --batch_size ${BATCH_SIZE} \
                --optimizer ${OPTIMIZER} \
                --learning_rate ${LEARNING_RATE} \
                --num_epochs ${NUM_EPOCHS} \
                --weight_decay ${WEIGHT_DECAY} \
                --learning ${LEARNING} \
                --architecture ${ARCHITECTURE}
        done
    done
done
