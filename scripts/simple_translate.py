from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

MODEL_NAME = "facebook/m2m100_418M"

# 1. Load the tokenizer (text <-> token ids) and the model (the actual network).
#    The first run downloads the weights (~1.9 GB) and caches them.
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()  # inference mode: turns off dropout

# 2. M2M-100 is multilingual, so we must tell it the source language.
tokenizer.src_lang = "zh"

sentences = [
    "我喜欢学习机器翻译。",
    "今天天气很好，我们去公园散步吧。",
]

# 3. Text -> token ids, as PyTorch tensors ("pt"). Padding makes the sentences equal length.
inputs = tokenizer(sentences, return_tensors="pt", padding=True)
print("input_ids shape:", inputs["input_ids"].shape)
print("tokens of sentence 1:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# 4. Generate target token ids. M2M-100 is told the target language by forcing
#    the first generated token to be the English language tag.
outputs = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.get_lang_id("en"),
    num_beams=1,  # 1 = greedy decoding
    max_new_tokens=100,
)

# 5. Token ids -> text.
translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for src, hyp in zip(sentences, translations):
    print(f"{src}\n  -> {hyp}")
