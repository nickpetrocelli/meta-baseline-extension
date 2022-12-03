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

# mini
qsub -m abes -l gpus=1 -cwd -e test_errors/mini_5shot_cos_nocorr_err.txt -o test_results/mini_5shot_cos_nocorr_out.psv -N mini_5shot_cos_nocorr run_scripts/run_test_1gpu.sh configs/test/test_few_shot_mini_5shot_cos.yaml 5
qsub -m abes -l gpus=1 -cwd -e test_errors/mini_5shot_cos_corr_err.txt -o test_results/mini_5shot_cos_corr_out.psv -N mini_5shot_cos_corr run_scripts/run_test_1gpu_corrupted.sh configs/test/test_few_shot_mini_5shot_cos.yaml 5

sleep 1h

qsub -m abes -l gpus=1 -cwd -e test_errors/mini_5shot_sqr_nocorr_err.txt -o test_results/mini_5shot_sqr_nocorr_out.psv -N mini_5shot_sqr_nocorr run_scripts/run_test_1gpu.sh configs/test/test_few_shot_mini_5shot_sqr.yaml 5
qsub -m abes -l gpus=1 -cwd -e test_errors/mini_5shot_sqr_corr_err.txt -o test_results/mini_5shot_sqr_corr_out.psv -N mini_5shot_sqr_corr run_scripts/run_test_1gpu_corrupted.sh configs/test/test_few_shot_mini_5shot_sqr.yaml 5

sleep 1h

qsub -m abes -l gpus=1 -cwd -e test_errors/mini_20shot_cos_nocorr_err.txt -o test_results/mini_20shot_cos_nocorr_out.psv -N mini_20shot_cos_nocorr run_scripts/run_test_1gpu.sh configs/test/test_few_shot_mini_20shot_cos.yaml 20
qsub -m abes -l gpus=1 -cwd -e test_errors/mini_20shot_cos_corr_err.txt -o test_results/mini_20shot_cos_corr_out.psv -N mini_20shot_cos_corr run_scripts/run_test_1gpu_corrupted.sh configs/test/test_few_shot_mini_20shot_cos.yaml 20

sleep 1h

qsub -m abes -l gpus=1 -cwd -e test_errors/mini_20shot_sqr_nocorr_err.txt -o test_results/mini_20shot_sqr_nocorr_out.psv -N mini_20shot_sqr_nocorr run_scripts/run_test_1gpu.sh configs/test/test_few_shot_mini_20shot_sqr.yaml 20
qsub -m abes -l gpus=1 -cwd -e test_errors/mini_20shot_sqr_corr_err.txt -o test_results/mini_20shot_sqr_corr_out.psv -N mini_20shot_sqr_corr run_scripts/run_test_1gpu_corrupted.sh configs/test/test_few_shot_mini_20shot_sqr.yaml 20

sleep 1h

# tiered
qsub -m abes -l gpus=1 -cwd -e test_errors/tiered_5shot_cos_nocorr_err.txt -o test_results/tiered_5shot_cos_nocorr_out.psv -N tiered_5shot_cos_nocorr run_scripts/run_test_1gpu.sh configs/test/test_few_shot_tiered_5shot_cos.yaml 5
qsub -m abes -l gpus=1 -cwd -e test_errors/tiered_5shot_cos_corr_err.txt -o test_results/tiered_5shot_cos_corr_out.psv -N tiered_5shot_cos_corr run_scripts/run_test_1gpu_corrupted.sh configs/test/test_few_shot_tiered_5shot_cos.yaml 5

sleep 1h

qsub -m abes -l gpus=1 -cwd -e test_errors/tiered_5shot_sqr_nocorr_err.txt -o test_results/tiered_5shot_sqr_nocorr_out.psv -N tiered_5shot_sqr_nocorr run_scripts/run_test_1gpu.sh configs/test/test_few_shot_tiered_5shot_sqr.yaml 5
qsub -m abes -l gpus=1 -cwd -e test_errors/tiered_5shot_sqr_corr_err.txt -o test_results/tiered_5shot_sqr_corr_out.psv -N tiered_5shot_sqr_corr run_scripts/run_test_1gpu_corrupted.sh configs/test/test_few_shot_tiered_5shot_sqr.yaml 5

sleep 1h

qsub -m abes -l gpus=1 -l gmem=24 -cwd -e test_errors/tiered_20shot_cos_nocorr_err.txt -o test_results/tiered_20shot_cos_nocorr_out.psv -N tiered_20shot_cos_nocorr run_scripts/run_test_1gpu.sh configs/test/test_few_shot_tiered_20shot_cos.yaml 20
qsub -m abes -l gpus=1 -l gmem=24 -cwd -e test_errors/tiered_20shot_cos_corr_err.txt -o test_results/tiered_20shot_cos_corr_out.psv -N tiered_20shot_cos_corr run_scripts/run_test_1gpu_corrupted.sh configs/test/test_few_shot_tiered_20shot_cos.yaml 20

sleep 1h

qsub -m abes -l gpus=1 -l gmem=24 -cwd -e test_errors/tiered_20shot_sqr_nocorr_err.txt -o test_results/tiered_20shot_sqr_nocorr_out.psv -N tiered_20shot_sqr_nocorr run_scripts/run_test_1gpu.sh configs/test/test_few_shot_tiered_20shot_sqr.yaml 20
qsub -m abes -l gpus=1 -l gmem=24 -cwd -e test_errors/tiered_20shot_sqr_corr_err.txt -o test_results/tiered_20shot_sqr_corr_out.psv -N tiered_20shot_sqr_corr run_scripts/run_test_1gpu_corrupted.sh configs/test/test_few_shot_tiered_20shot_sqr.yaml 20

