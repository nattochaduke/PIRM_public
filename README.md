# PIRM_public
This is a code for PIRM 2018 Spectral Image Super-Resolution Challenge.

## Setting:
- Dockerfiles we used to build our environments are provided in the directory *dockerfiles*. We provided 2 dockerfiles, 1 for cuda 9.1 and another for cuda 9.2.
- Extract and save all training, validation, testing data in *test* directory according with the names of sub-directories. Running *data/preprocess.py* generate normalized data.

## Training
We conduct 2 training phases; training from scratch with MAE loss and finetuning with weighted sum of MAE, MRAE, SID loss.
### Track 1
1. To conduct training from scratch in track 1, execute "python3 t1_densetraining.py -b 4 --learning_rate 3e-4 --loss_coeffs 1,0,0,0 --image_concat 1 --res_scale 1 --last_relu False --calc_sid False --n_RDBs 20 --n_feats 256 --growth_rate 64"  
This will save the training process and snapshots in training in *result_t1* directory. Choose the smallest Mean Absolute Error snapshot and make the name of it **t1_trained**.  
2. Then, execute "python3 t1_densetraining.py --out t1_finetuned -b 4 --learning_rate 3e-5 --loss_coeffs 5,0,0.1,0.0001 --image_concat 1  --res_scale 1 --last_relu True --calc_sid True --n_RDBs 20 --n_feats 256 --growth_rate 64 --resume **t1_trained**"
This will save the training process and snapshots in training in *finetuned_t1* directory. Choose the smallest MRAE snapshot and make it to be the final model for track 1. Let it be *model_track1*

### Track 2
1. To conduct training from scratch in track 2, execute "python3 t2_densetraining.py -b 2 --learning_rate 3e-4 --loss_coeffs 1,0,0,0 --image_concat 1 --flip False --rotate False --res_scale 1 --last_relu False --calc_sid False --n_RDBs 20 --n_feats 256 --growth_rate 64"
This will save the training process and snapshots in training in *result_t2* directory. Choose the smallest MAE Error snapshot and let the name of it **t2_trained**.  
2. Then, execute "python3 t2_densetraining.py --out t2_finetuned -b 4 --learning_rate 3e-5 --loss_coeffs 5,0,0.1,0.0001 --image_concat 1 --flip False --rotate False --res_scale 1 --last_relu True --calc_sid True --n_RDBs 20 --n_feats 256 --growth_rate 64 --resume **t2_trained**".
This will save the training process and snapshots in training in *finetuned_t2* directory. Choose the smallest MRAE snapshot and make it to be the final model for track 2. Let it be *model_track2*.

## Inference
1. To conduct inference in track 1, execute "python3 inference_t1.py --model *model_track1* --target result_track1". This will generate result zip file in result_track1 directory.
2. To conduct inference in track 1, execute "python3 inference_t2.py --model *model_track2* --target result_track2". This will generate result zip file in result_track2 directory.
