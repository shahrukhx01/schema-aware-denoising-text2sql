## transformers-cli login
## !apt-get install git-lfs
## git config --global user.email "sk28671@gmail.com"
from transformers import (
    CONFIG_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)

model = AutoModelForSeq2SeqLM.from_pretrained("./output/model_2")
model.push_to_hub("schema-aware-denoising-distilbart-cnn-12-6-text2sql")

tokenizer = AutoTokenizer.from_pretrained("./output/model_2")
tokenizer.push_to_hub("schema-aware-denoising-distilbart-cnn-12-6-text2sql")
