#!/bin/bash


DATASET=cifar-100
SEEDS=(0 1 2 3 4)
EPOCHS=100
BATCH_SIZE=64
NUM_WORKERS=8
OPTIMIZER=custom_sgd

MODEL=haframe_wide_resnet
LOSS=cross-entropy

GAMMA=1.0
ALPHA=0.05

for seed in "${SEEDS[@]}";
do
    output_dir=out/${DATASET}/hier-cos/alpha_${ALPHA}/gamma_${GAMMA}/${ALPHA}-${LOSS}-${GAMMA}-${MODEL}-seed_${seed}

  # train
  python main.py \
  --start training \
  --arch ${MODEL} \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --loss ${LOSS} \
  --loss-schedule ${ALPHA} \
  --haf-gamma ${GAMMA} \
  --optimizer ${OPTIMIZER} \
  --data ${DATASET} \
  --workers ${NUM_WORKERS} \
  --output "${output_dir}" \
  --seed "${seed}" \
  --feature_space "hier-cos" \
  --alpha ${ALPHA}

  # test
  python main.py \
  --start testing \
  --arch ${MODEL} \
  --loss ${LOSS} \
  --loss-schedule ${ALPHA} \
  --haf-gamma ${GAMMA} \
  --optimizer ${OPTIMIZER} \
  --data ${DATASET} \
  --workers ${NUM_WORKERS} \
  --output "${output_dir}" \
  --seed "${seed}" \
  --feature_space "hier-cos" \
  --alpha ${ALPHA}

done

# 5 runs evaluation on baseline model
python experiments/multiple_runs_eval.py --arch ${MODEL} --loss ${ALPHA}-${LOSS}-${GAMMA} \
--nseed ${#SEEDS[@]} --output out/${DATASET}/hier-cos/alpha_${ALPHA}/gamma_${GAMMA}/

