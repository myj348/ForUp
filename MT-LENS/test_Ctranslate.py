from ctranslate2 import Translator
import transformers

model_dir = "./ctranslate2/opus-mt-zh-en-ctranslate2" # Path to model directory.
translator = Translator(
            model_path=model_dir,
            device="cpu", # cpu, cuda, or auto.
            inter_threads=1, # Maximum number of parallel translations.
            intra_threads=4, # Number of OpenMP threads per translator.
            compute_type="int8", # int8 for cpu or int8_float16 for cuda.
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)

source = tokenizer.convert_ids_to_tokens(tokenizer.encode("怎么老是你？添加了更具描述性的打印语句"))
results = translator.translate_batch([source])
target = results[0].hypotheses[0]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
