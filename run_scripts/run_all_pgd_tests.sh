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


#qsub -m abes -l gpus=1 -cwd -e test_errors/mini_5shot_pgd_nocorr_err.txt -o test_results/mini_5shot_pgd_nocorr_out.psv -N mini_5shot_pgd_nocorr run_scripts/run_test_1gpu_pgd.sh configs/test/test_few_shot_mini_5shot_sqr.yaml 5
# sleep 30m
#qsub -m abes -l gpus=1 -cwd -e test_errors/mini_5shot_pgd_corr_err.txt -o test_results/mini_5shot_pgd_corr_out.csv -N mini_5shot_pgd_corr run_scripts/run_test_1gpu_pgd_corrupted.sh configs/test/test_few_shot_mini_5shot_sqr.yaml 5

sleep 30m

qsub -m abes -l gpus=1 -l gmem=11 -cwd -e test_errors/mini_20shot_pgd_nocorr_err.txt -o test_results/mini_20shot_pgd_nocorr_out.csv -N mini_20shot_pgd_nocorr run_scripts/run_test_1gpu_pgd.sh configs/test/test_few_shot_mini_20shot_sqr.yaml 20
sleep 3h
qsub -m abes -l gpus=1 -l gmem=11 -cwd -e test_errors/mini_20shot_pgd_corr_err.txt -o test_results/mini_20shot_pgd_corr_out.csv -N mini_20shot_pgd_corr run_scripts/run_test_1gpu_pgd_corrupted.sh configs/test/test_few_shot_mini_20shot_sqr.yaml 20

sleep 3h

# tiered


qsub -m abes -l gpus=1 -l gmem=11 -cwd -e test_errors/tiered_5shot_pgd_nocorr_err.txt -o test_results/tiered_5shot_pgd_nocorr_out.csv -N tiered_5shot_pgd_nocorr run_scripts/run_test_1gpu_pgd.sh configs/test/test_few_shot_tiered_5shot_sqr.yaml 5
sleep 3h
qsub -m abes -l gpus=1 -l gmem=11 -cwd -e test_errors/tiered_5shot_pgd_corr_err.txt -o test_results/tiered_5shot_pgd_corr_out.csv -N tiered_5shot_pgd_corr run_scripts/run_test_1gpu_pgd_corrupted.sh configs/test/test_few_shot_tiered_5shot_sqr.yaml 5

sleep 3h


qsub -m abes -l gpus=1 -l gmem=11 -cwd -e test_errors/tiered_20shot_pgd_nocorr_err.txt -o test_results/tiered_20shot_pgd_nocorr_out.csv -N tiered_20shot_pgd_nocorr run_scripts/run_test_1gpu_pgd.sh configs/test/test_few_shot_tiered_20shot_sqr.yaml 20
sleep 3h
qsub -m abes -l gpus=1 -l gmem=11 -cwd -e test_errors/tiered_20shot_pgd_corr_err.txt -o test_results/tiered_20shot_pgd_corr_out.csv -N tiered_20shot_pgd_corr run_scripts/run_test_1gpu_pgd_corrupted.sh configs/test/test_few_shot_tiered_20shot_sqr.yaml 20

