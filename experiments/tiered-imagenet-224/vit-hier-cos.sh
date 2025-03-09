#!/bin/bash


DATASET=tiered-imagenet-224
SEEDS=(0 1 2 3 4)
EPOCHS=10
BATCH_SIZE=256
NUM_WORKERS=8
OPTIMIZER=custom_sgd

MODEL=haframe_vit
POOLING=average
LOSS=cross-entropy

GAMMA=1.0 # Gamma = 1.0 for inaturalist
ALPHA=0.01

for seed in "${SEEDS[@]}";
do
  output_dir=out/${DATASET}/vit-hier-cos/alpha_${ALPHA}/gamma_${GAMMA}/${ALPHA}-${LOSS}-${GAMMA}-${MODEL}-seed_${seed}

  # train
  python main.py \
  --start training \
  --arch ${MODEL} \
  --pool ${POOLING} \
  --larger-backbone \
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
  --pool ${POOLING} \
  --larger-backbone \
  --loss ${LOSS} \
  --loss-schedule ${ALPHA} \
  --haf-gamma ${GAMMA} \
  --optimizer ${OPTIMIZER} \
  --data ${DATASET} \
  --workers ${NUM_WORKERS} \
  --output "${output_dir}" \
  --seed "${seed}" \
  --feature_space "hier-cos"

done

# 5 runs evaluation on baseline model
python experiments/multiple_runs_eval.py --arch ${MODEL} --loss ${ALPHA}-${LOSS}-${GAMMA} \
--nseed ${#SEEDS[@]} --output out/${DATASET}/vit-hier-cos/alpha_${ALPHA}/gamma_${GAMMA}

