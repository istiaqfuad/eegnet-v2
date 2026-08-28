#!/bin/bash
# E2: IV-2a within-subject, both protocols (A: subject-dependent pure, B: transfer-assisted)
#     arms: none / RA / RA+IM-TTA, seeds 42/1/2
# E3: IV-2b grid — within (transfer) + LOSO, arms none/EA/RA x off/IM, seeds 42/1/2
# All runs: unified normalization, dropout-free TTA, --clean (no stale pretrain cache).
# Concurrency: 5 streams (seeds sequential inside each), num_workers=1 — shares CPU with E1.
cd /root/eegnet-v2
NW=1

run () {  # run <dataset> <protocol> <align> <ttaargs> <seed> <tag>
  local DS=$1 PROTO=$2 ALIGN=$3 TTAARGS=$4 SEED=$5 TAG=$6
  local PAT="results_*_${DS}_unified_${TAG}.csv"
  if ls $PAT >/dev/null 2>&1; then
    echo "skip $TAG (exists)"; return
  fi
  python main.py --protocol $PROTO --model unified --dataset $DS --num_workers $NW \
    --align $ALIGN $TTAARGS --seed $SEED --tag $TAG --clean \
    > logs_v2/${TAG}.log 2>&1
  echo "done $TAG"
}

# S1: E2a — IV-2a pure subject-dependent (protocol A)
(
  for SEED in 42 1 2; do
    run iv2a within_pure none ""                            $SEED "v2pure_none_off_s${SEED}"
    run iv2a within_pure ra   ""                            $SEED "v2pure_ra_off_s${SEED}"
    run iv2a within_pure ra   "--tta_steps 5 --tta_div 1.0" $SEED "v2pure_ra_im_s${SEED}"
  done
) &

# S2: E2b — IV-2a transfer-assisted (protocol B)
(
  for SEED in 42 1 2; do
    run iv2a within      none ""                            $SEED "v2trans_none_off_s${SEED}"
    run iv2a within      ra   ""                            $SEED "v2trans_ra_off_s${SEED}"
    run iv2a within      ra   "--tta_steps 5 --tta_div 1.0" $SEED "v2trans_ra_im_s${SEED}"
  done
) &

# S3: E3 — IV-2b within (transfer-assisted)
(
  for SEED in 42 1 2; do
    run iv2b within none ""                                  $SEED "v2b_within_none_off_s${SEED}"
    run iv2b within ra   ""                                  $SEED "v2b_within_ra_off_s${SEED}"
    run iv2b within ra   "--tta_steps 5 --tta_div 1.0"       $SEED "v2b_within_ra_im_s${SEED}"
  done
) &

# S4: E3 — IV-2b LOSO main grid
(
  for SEED in 42 1 2; do
    run iv2b loso none ""                                    $SEED "v2b_loso_none_off_s${SEED}"
    run iv2b loso ra   ""                                    $SEED "v2b_loso_ra_off_s${SEED}"
    run iv2b loso ra   "--tta_steps 5 --tta_div 1.0"         $SEED "v2b_loso_ra_im_s${SEED}"
  done
) &

# S5: E3 — IV-2b LOSO ablation arms (alignment x TTA matrix completion)
(
  for SEED in 42 1 2; do
    run iv2b loso none "--tta_steps 5 --tta_div 1.0"         $SEED "v2b_loso_none_im_s${SEED}"
    run iv2b loso ea   ""                                    $SEED "v2b_loso_ea_off_s${SEED}"
    run iv2b loso ea   "--tta_steps 5 --tta_div 1.0"         $SEED "v2b_loso_ea_im_s${SEED}"
  done
) &

wait
echo "E2/E3 COMPLETE"
