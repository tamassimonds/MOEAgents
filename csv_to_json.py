# csv_to_json.py

import csv
import json
import random
from typing import List, Dict, Any

def format_question(row: Dict[str, str]) -> str:
    question = row['question']
    choices = eval(row['choices'])  # Convert string representation of list to actual list
    formatted_choices = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
    return f"{question}\n\nAnswer Choices:\n{formatted_choices}"

def index_to_letter(index: str) -> str:
    return chr(65 + int(index))

def generate_json_dataset(subject: str, csv_file: str = 'mmlu_dataset_with_domains.csv', num_questions: int = None) -> List[Dict[str, str]]:
    formatted_data = []
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            if row['domain'] == subject:
                formatted_question = {
                    "instruction": format_question(row),
                    "output": f"The correct answer is {index_to_letter(row['answer'])}"
                }
                formatted_data.append(formatted_question)
    
    # Shuffle the formatted data
    random.shuffle(formatted_data)
    
    # Limit the number of questions if specified
    if num_questions is not None:
        formatted_data = formatted_data[:num_questions]
    
    # Write the shuffled formatted data to a JSON file
    output_file = f'formatted_{subject.lower()}_questions.json'
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(formatted_data, json_file, indent=2, ensure_ascii=False)
    
    print(f"Processed and shuffled {len(formatted_data)} {subject} questions and saved to '{output_file}'")
    
    return formatted_data

if __name__ == "__main__":
    # Example usage
    subject = "Computer Science"
    generate_json_dataset(subject, num_questions=50)