from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", use_fast=False)

# 保存分词器文件到指定目录
tokenizer.save_pretrained("./ctranslate2/opus-mt-zh-en-ctranslate2/")

