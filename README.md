# SeaD: End-to-end Text-to-SQL Generation with Schema-aware Denoising with external fuzzy memory lookup
Unofficial implementation of [SeaD: End-to-end Text-to-SQL Generation with Schema-aware Denoising](https://www.arxiv-vanity.com/papers/2105.07911/) with some potential ehancements for industrial use cases namely external fuzzy memory lookup.
```bash
python finetuning_bart.py --model_name_or_path='shahrukhx01/schema-aware-denoising-distilbart-cnn-12-6-text2sql' --dataset_name='wikisql_augmented_data' --per_device_train_batch_size=4 --output_dir=output --num_train_epochs=20
```
