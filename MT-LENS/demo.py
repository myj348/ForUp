from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# 输入文本
text = "你好，世界！"

# 分词
inputs = tokenizer(text, return_tensors="pt")

# 生成翻译
outputs = model.generate(**inputs)

# 解码输出
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translated_text)
