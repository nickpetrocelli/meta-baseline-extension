#!/bin/bash
#
#  Execute from the current working directory
# -cwd
#
#  This is a long-running job
# -l inf
#
#
#
# notify
# -m abes
#
qsub -l gpus=1 -l gmem=24 -cwd -e test_errors/mini_5shot_cos_nocorr_err.txt -o test_results/mini_5shot_cos_nocorr_out.psv -N mini_5shot_cos_nocorr run_scripts/run_test_1gpu.sh configs/test/test_few_shot_mini_5shot_cos.yaml 5
qsub -l gpus=1 -l gmem=24 -cwd -e test_errors/mini_5shot_cos_corr_err.txt -o test_results/mini_5shot_cos_corr_out.psv -N mini_5shot_cos_corr run_scripts/run_test_1gpu_corrupted.sh configs/test/test_few_shot_mini_5shot_cos.yaml 5

