
for env in "PandaPickAndPlace-v3" "PandaSlide-v3" "PandaReach-v3"
do
        for seed in 4 5 6 7 8
        do
                python estimate_real_dis_panda.py \
                        --debug 0 \
                        --folder "exp/panda" \
                        --algo "tqc" \
                        --env $env \
                        --policy "gail_estimate" \
                        --seed $seed \
                        --exp_name "seed${seed}" \
                        --exp_dir "si" \
                        --bc_budget 100000 \
                        --bc_batch_size 1024 \
                        --bc_num_epochs 5 \
                        --retrain_bc_epochs 2000 \
                        --estimate_dist "real_dist_shift" \
                        --monitor 1 \
                        --adapting_bc_budget 1 \
                        --last_bc_budget 1000000 \
                        --check_list 5000000 10000000 15000000 20000000 25000000 30000000 35000000 40000000 45000000 50000000 \
                        --last_lr_decay 0 \
                        --early_stopping 1 \
                        --eval_freq_epoch 200
        done
done