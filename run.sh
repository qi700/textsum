#先采用ml进行训练
python train.py --train_mle=yes --train_rl=no --mle_weight=1.0
#模型评估
#python decode.py --task=validate --start_from=model_50.tar  还没好
python3.7 e-eval.py --task=validate --start_from=model_50.tar
#测试可以按照外面的代码的逻辑进行修改

#ml和rl一起训练，选择一个效果最好的模型进行继续训练
python train.py --train_mle=yes --train_rl=yes --mle_weight=0.25 --load_model=model_100.tar --new_lr=0.0001