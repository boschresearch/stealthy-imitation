for env in "PandaPickAndPlace-v3" "PandaSlide-v3" "PandaReach-v3" 
do
        python -u real_env_analize_panda.py \
                --folder "exp/panda" \
                --algo "tqc" \
                --env $env \
                --seed 0 \
                --n_sample 1000000 \
                --visualize 0

done