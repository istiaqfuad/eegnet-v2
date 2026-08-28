#!/bin/bash
# E1: IV-2a LOSO matrix {none,ea,ra} x {off,tent,im} x seeds {42,1,2}, dataloader workers=2
cd /root/eegnet-v2
for ALIGN in none ea ra; do
  for TTA in off tent im; do
    (
      for SEED in 42 1 2; do
        TAG="v2_${ALIGN}_${TTA}_s${SEED}"
        if [ -f "results_loso_iv2a_unified_${TAG}.csv" ]; then
          echo "skip $TAG (exists)"; continue
        fi
        if [ "$TTA" = "off" ]; then TTAARGS="";
        elif [ "$TTA" = "tent" ]; then TTAARGS="--tta_steps 5 --tta_div 0.0";
        else TTAARGS="--tta_steps 5 --tta_div 1.0"; fi
        python main.py --protocol loso --model unified --dataset iv2a --num_workers 2 \
          --align $ALIGN $TTAARGS --seed $SEED --tag $TAG \
          > logs_v2/${TAG}.log 2>&1
        echo "done $TAG"
      done
    ) &
  done
done
wait
echo "E1 COMPLETE"
