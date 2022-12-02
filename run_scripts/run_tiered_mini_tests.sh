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
qsub -l gpus=1 -l gmem=24 -cwd -e test_errors/mini_5shot_cos_err.txt -o test_results/mini_5shot_cos_out.psv -N mini_5shot_cos run_scripts/run_test_1gpu.sh configs/test/test_few_shot_mini_5shot_cos.yaml 5

