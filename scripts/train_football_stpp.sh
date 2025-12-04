cd ../
dataset=football-stpp
model=SMASH
sigma_time=0.2
sigma_loc=0.1
samplingsteps=2000
langevin_step=0.01
log_normalization=1
batch_size=4
cond_dim=32
seed=617
loss_lambda=0.5

save_path=./ModelSave/stpp-dataset_${dataset}_model_${model}_sigma_${sigma_time}_${sigma_loc}_log_${log_normalization}_seed_${seed}_dim_${cond_dim}_lambda_${loss_lambda}/
python app.py --dim 2 --loss_lambda ${loss_lambda} --cond_dim ${cond_dim} --seed ${seed} --save_path ${save_path} --log_normalization ${log_normalization} --dataset ${dataset} --mode train --model ${model} --samplingsteps -1 --batch_size ${batch_size} --total_epochs 150 --n_samples 1 --per_step 1 --sigma_time ${sigma_time} --sigma_loc ${sigma_loc}

python app.py --loss_lambda ${loss_lambda} --cond_dim ${cond_dim} --seed ${seed} --dim 2 --log_normalization ${log_normalization} --dataset ${dataset} --mode test --model ${model} --langevin_step ${langevin_step} --batch_size ${batch_size} --samplingsteps ${samplingsteps} --n_samples 30 --per_step 100  --sigma_time ${sigma_time} --sigma_loc ${sigma_loc} --weight_path ${save_path}model_best.pkl


