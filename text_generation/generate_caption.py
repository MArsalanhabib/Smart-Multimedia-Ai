
from transformers import pipeline

captioner = pipeline("text-generation", model="gpt2")

def generate_caption(prompt):
    result = captioner(prompt, max_length=50, num_return_sequences=1)
    return result[0]['generated_text']
