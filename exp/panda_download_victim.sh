for env in "PandaPickAndPlace-v3" "PandaSlide-v3" "PandaReach-v3"
do
        python -m rl_zoo3.load_from_hub --algo tqc --env ${env} -orga chencliu -f exp/panda
done