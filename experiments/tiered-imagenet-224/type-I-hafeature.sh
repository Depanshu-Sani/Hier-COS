#!/bin/bash


DATASET=tiered-imagenet-224
SEEDS=(0 1 2 3 4)
EPOCHS=100
BATCH_SIZE=256
NUM_WORKERS=8
OPTIMIZER=custom_sgd

MODEL=custom_resnet50
LOSS=hafeat-l12-cejsd-wtconst-dissim

for seed in "${SEEDS[@]}";
do
  output_dir=out/${DATASET}/hafeat/${LOSS}-${MODEL}-seed_${seed}

  # train
  python main.py \
  --start training \
  --arch ${MODEL} \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --loss ${LOSS} \
  --optimizer ${OPTIMIZER} \
  --data ${DATASET} \
  --workers ${NUM_WORKERS} \
  --output "${output_dir}" \
  --seed "${seed}"

  # test
  python main.py \
  --start testing \
  --arch ${MODEL} \
  --loss ${LOSS} \
  --optimizer ${OPTIMIZER} \
  --data ${DATASET} \
  --workers ${NUM_WORKERS} \
  --output "${output_dir}" \
  --seed "${seed}"

done

# 5 runs evaluation on baseline model
python experiments/multiple_runs_eval.py --arch ${MODEL} --loss ${LOSS} \
--nseed ${#SEEDS[@]} --output out/${DATASET}/hafeat/

