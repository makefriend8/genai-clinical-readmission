import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_longt5_model(model_name="google/long-t5-tglobal-base"):
    """
    Load the LongT5 model for conditional generation.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model

def get_longt5_tokenizer(model_name="google/long-t5-tglobal-base"):
    """
    Load the corresponding tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")
    return tokenizer

if __name__ == "__main__":
    print("Loading LongT5 model and tokenizer to test...")
    tokenizer = get_longt5_tokenizer()
    model = get_longt5_model()
    print("LongT5 architecture successfully loaded!")
    print(f"Model vocab size: {model.config.vocab_size}")
