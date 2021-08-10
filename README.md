# finetune_bart_seq2seq

`bash
nohup python finetuning_bart.py --model_name_or_path='sshleifer/distilbart-cnn-12-6' --dataset_name='procurement_data' --per_device_train_batch_size=8 --output_dir=output &
`
