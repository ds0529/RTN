## Part segmentation

### Data prepare

Download the ShapenetPart dataset:

```
sh download_data.sh
```

, and put the folders `shapenet_hdf5_data` and `shapenet_PartAnnotation` to `$your_data_folder`.

### Train

Train the RTN. 

```
python train_learning_rotation_so3.py --gpu $your_gpu_id --data_path $your_data_folder --log_dir $your_RTN_model_path
```

Train the RTN+DGCNN with multi GPUs. 

```
python train_multi_gpu_rotation_pretrain_so3.py --data_path $your_data_folder --transformer_model_path $your_RTN_model_path --output_dir $your_model_path
```

### Test

Test the RTN. 

```
python evaluate_learning_rotation_so3.py --gpu $your_gpu_id --data_path $your_data_folder --model_path $your_RTN_model_path --dump_dir $your_RTN_test_path
```

Test the RTN+DGCNN. 

```
python test_rotation_pretrain_so3.py --data_path $your_data_folder --transformer_model_path $your_RTN_model_path --model_path $your_model_path --dump_dir $your_test_path
```

The trained models are stored in the folder `result`.