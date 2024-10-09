import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from tqdm import tqdm
import logging
from datetime import datetime

from lib.classify import classify_text

# Load category mapping
with open("category_mapping.json", "r") as f:
    category_mapping = json.load(f)

# Define the list of domains
domains = ["Math", "STEM", "Coding", "Other"]

# Set up logging
log_filename = f"classification_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

# Function to map domain to category
def map_domain(domain):
    if domain in domains:
        return domain
    return "Other"

# Load MMLU dataset
mmlu_df = pd.read_csv("mmlu_dataset_with_domains.csv")

# Function to benchmark classifier on n examples
def benchmark_classifier(n):
    correct = 0
    total = 0
    
    for _, row in tqdm(mmlu_df.sample(n=n).iterrows(), total=n):
        question = row['question']
        true_domain = map_domain(row['domain'])
        
        predicted_category = classify_text(question)
        
        # Log the example
        logging.info(f"Question: {question}")
        logging.info(f"True Domain: {true_domain}")
        logging.info(f"Predicted Category: {predicted_category}")
        logging.info(f"Correct: {predicted_category == true_domain}\n")
        
        if predicted_category == true_domain:
            correct += 1
        total += 1
        
    accuracy = correct / total
    return accuracy

# Run benchmark
n_examples = 100  # Change this to the desired number of examples
accuracy = benchmark_classifier(n_examples)
print(f"Accuracy on {n_examples} examples: {accuracy:.2f}")
