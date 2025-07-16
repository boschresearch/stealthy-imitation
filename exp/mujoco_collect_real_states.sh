for env in hopper walker2d halfcheetah
do
        python real_env_analize.py \
                --policy_path "exp/${env}/sac_seed0" \
                --n_sample 1000000
done