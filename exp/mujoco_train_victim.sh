for env in hopper walker2d halfcheetah
do
    python train_victim.py \
            --env ${env} \
            --policy sac \
            --log_obs_boundary 1
done