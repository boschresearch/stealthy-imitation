for env in hopper walker2d halfcheetah
do
        for seed in 4 5 6 7 8
        do
                python estimate_real_dis.py \
                        --debug 0 \
                        --victim_cfg "exp/${env}/sac_seed0" \
                        --env $env \
                        --seed $seed \
                        --exp_name "seed${seed}" \
                        --exp_dir "si" \
                        --bc_budget 100000 \
                        --bc_batch_size 1024 \
                        --retrain_bc_epochs 2000 \
                        --monitor 1 \
                        --adapting_bc_budget 1 \
                        --last_bc_budget 1000000 \
                        --check_list 10000000 30000000 50000000 \
                        --last_lr_decay 0 \
                        --early_stopping 1
        done
done