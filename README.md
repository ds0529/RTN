## Rotation Transformation Network: Learning View-Invariant Point Cloud for Classification and Segmentation

### Align ModelNet40

Align the ModelNet40 dataset, and put the generated folder 'aligned_modelnet40_ply_hdf5_2048' to $your_data_folder

```
cd align_modelnet
python data_prepare.py
```

### Train

Train the RTN. 

```
python train_learning_rotation_so3.py --gpu $your_gpu_id --data_path $your_data_folder --log_dir $your_RTN_model_path
```

Train the RTN+DGCNN. 

```
python train_rotation_pretrain_so3.py --gpu $your_gpu_id --data_path $your_data_folder --transformer_model_path $your_RTN_model_path --log_dir $your_model_path
```

### Test

Test the RTN. 

```
python evaluate_learning_rotation_so3.py --gpu $your_gpu_id --data_path $your_data_folder --model_path $your_RTN_model_path --dump_dir $your_RTN_test_path
```

Test the RTN+DGCNN. 

```
python evaluate_rotation_pretrain_so3.py --gpu $your_gpu_id --data_path $your_data_folder --transformer_model_path $your_RTN_model_path --model_path $your_model_path --dump_dir $your_test_path
```