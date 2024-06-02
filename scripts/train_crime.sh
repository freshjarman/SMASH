cd ../
dataset=crime
model=SMASH
sigma_time=0.3
sigma_loc=0.03
samplingsteps=1000
langevin_step=0.01
log_normalization=1
seed=4
smooth=0.1
loss_lambda=0.5

save_path=./ModelSave/dataset_${dataset}_model_${model}_sigma_${sigma_time}_${sigma_loc}_log_${log_normalization}_seed_${seed}_lambda_${loss_lambda}/
python app.py --dim 3 --loss_lambda ${loss_lambda} --save_path ${save_path} --seed ${seed} --smooth ${smooth} --log_normalization ${log_normalization} --dataset ${dataset} --mode train --model ${model} --samplingsteps -1 --batch_size 32 --total_epochs 150 --n_samples 1 --per_step 1 --sigma_time ${sigma_time} --sigma_loc ${sigma_loc}

python app.py --dim 3 --loss_lambda ${loss_lambda} --smooth ${smooth} --log_normalization ${log_normalization} --dataset ${dataset} --mode test --model ${model} --langevin_step ${langevin_step} --samplingsteps ${samplingsteps} --n_samples 30 --per_step 250  --sigma_time ${sigma_time} --sigma_loc ${sigma_loc} --weight_path ${save_path}model_best.pkl


