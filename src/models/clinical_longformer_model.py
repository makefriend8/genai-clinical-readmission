import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_clinical_longformer_classifier(model_name="yikuan8/Clinical-Longformer", num_labels=2):
    """
    Load the Clinical-Longformer model for downstream 30-day readmission classification.
    Used in Experiment 1 & 3 to evaluate the predictive signal retention of the generated summaries.
    """
    # We initialize it for sequence classification (binary: readmitted or not)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # As per our experimental design, we freeze the base model weights 
    # and only use it as a fixed feature extractor / train the classification head
    # to ensure a fair comparison between Raw Notes, Human BHC, and Machine BHC.
    for param in model.longformer.parameters():
        param.requires_grad = False
        
    return model

def get_clinical_longformer_tokenizer(model_name="yikuan8/Clinical-Longformer"):
    """
    Load the corresponding Clinical-Longformer tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

if __name__ == "__main__":
    print("Loading Clinical-Longformer classifier and tokenizer to test...")
    tokenizer = get_clinical_longformer_tokenizer()
    model = get_clinical_longformer_classifier()
    print("Clinical-Longformer architecture successfully loaded (base weights frozen)!")
