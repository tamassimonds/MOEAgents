import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Define paths for local storage
MODEL_PATH = "local_model"
TOKENIZER_PATH = "local_tokenizer"

#hubspot_model = "TamasSimonds/bert_domain_classifier"
hubspot_model = "TamasSimonds/domain_classifier"
# Load or download model and tokenizer
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    print("Loading model and tokenizer from local storage...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    print("Downloading model and tokenizer...")

    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    model = AutoModelForSequenceClassification.from_pretrained(hubspot_model)
    
    # Save model and tokenizer locally
    print("Saving model and tokenizer to local storage...")
    tokenizer.save_pretrained(TOKENIZER_PATH)
    model.save_pretrained(MODEL_PATH)

# Prepare your text
text = "what is the radius of the earth?"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Make prediction with probabilities
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    probabilities_dict = {i: prob.item() for i, prob in enumerate(probabilities[0])}

# Load category mapping
import json
with open("category_mapping.json", "r") as f:
    category_mapping = json.load(f)

# Function to classify text with probabilities
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_category = list(category_mapping.keys())[list(category_mapping.values()).index(predicted_class)]
    probabilities_dict = {i: prob.item() for i, prob in enumerate(probabilities[0])}

    return predicted_category