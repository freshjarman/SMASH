cd ../
device=0
dataset=Earthquake
model=SMASH
sigma_time=0.2
sigma_loc=0.25
samplingsteps=2000
langevin_step=0.005
log_normalization=1
cond_dim=16
seed=2
loss_lambda=0.5

save_path=./ModelSave/dataset_${dataset}_model_${model}_sigma_${sigma_time}_${sigma_loc}_log_${log_normalization}_seed_${seed}_dim_${cond_dim}_lambda_${loss_lambda}/
python app.py --loss_lambda ${loss_lambda} --dim 3 --cond_dim ${cond_dim} --save_path ${save_path} --seed ${seed} --log_normalization ${log_normalization} --dataset ${dataset} --mode train --model ${model} --samplingsteps -1 --batch_size 32 --total_epochs 150 --n_samples 1 --per_step 1 --sigma_time ${sigma_time} --sigma_loc ${sigma_loc}
python app.py --loss_lambda ${loss_lambda} --cond_dim ${cond_dim} --seed ${seed} --dim 3 --log_normalization ${log_normalization} --dataset ${dataset} --mode test --model ${model} --langevin_step ${langevin_step} --samplingsteps ${samplingsteps} --n_samples 30 --per_step 250  --sigma_time ${sigma_time} --sigma_loc ${sigma_loc} --weight_path ${save_path}model_best.pkl


