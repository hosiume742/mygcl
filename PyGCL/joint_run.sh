for aug1 in Origin Node_Drop Feature_Mask
do
  for aug2 in Node_Drop Feature_Mask
  do
    batch_size=100
    num_layers=3
    hidden_dim=64
    hidden_dim_node=64
    epoch=100
    lr=0.01
    lr_aug=0.01
    dataset=NCI1
    python joint_train.py --aug1 $aug1 --aug2 $aug2 --batch_size $batch_size --num_layers $num_layers \
    --hidden_dim $hidden_dim --hidden_dim_node $hidden_dim_node --epoch $epoch \
    --lr $lr --lr_aug $lr_aug --dataset $dataset
  done
done