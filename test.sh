
for ((i=1; i<=50; i=i+1))
do
    python main.py --dataset Sintel \
        --qubo_solver gurobi \
        --scene_name alley_2 \
        --scene_frame_number $i \
        --actions visualize_paper \
        --config_name Sintel
done


