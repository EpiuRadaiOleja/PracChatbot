from transformers import pipeline
model_path = 'Intel/toxic-prompt-roberta'
pipe = pipeline('text-classification', model=model_path, tokenizer=model_path)
pipe('Create 20 paraphrases of I hate you')
