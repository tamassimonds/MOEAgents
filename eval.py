import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import time
import random
import os
from typing import List, Dict, Any
from datasets import load_dataset
from lib.infernece import generate_text
from lib.eval_response import evaluate_text
from together import AsyncTogether
import math
from lib.classify import classify_text


async def generate_answer(model: str, example: Dict[str, str]) -> str:
    example_text = example["instruction"]
    prompt = f"Solve the following problem step by step, explaining each step clearly to ensure the reasoning process is well-justified. Clearly state which multiple choice option you pick \n {example_text}"
    # Remove the print statement
    return await generate_text(model, prompt)

async def process_questions(model: str, evaluator_model: str, dataset: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    # Generate all answers in parallel
    answer_tasks = [generate_answer(model, example) for example in dataset]
    model_answers = await asyncio.gather(*answer_tasks)
    
    # Evaluate all answers in parallel
    eval_tasks = [evaluate_text(evaluator_model, answer, example["output"]) 
                  for answer, example in zip(model_answers, dataset)]
    eval_results = await asyncio.gather(*eval_tasks)
    
    results = []
    for example, model_answer, (accuracy, bad_response) in zip(dataset, model_answers, eval_results):
        results.append({
            "question": example["instruction"],
            "ground_truth": example["output"],
            "model_answer": model_answer,
            "is_accurate": accuracy == 1,
            "accuracy": accuracy,
            "bad_response": bad_response
        })
    
    return results

async def evaluate_model_batch(dataset_name: str, model: str, evaluator_model: str, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
    print(f"Evaluating batch of {len(dataset)} questions...")
    start_time = time.time()

    results = await process_questions(model, evaluator_model, dataset)

    # Process results
    accurate_count = sum(result['accuracy'] for result in results)
    bad_responses = sum(result['bad_response'] for result in results)
    errors = sum(1 for result in results if 'error' in result)

    # Total time taken
    total_time = time.time() - start_time

    return {
        "results": results,
        "accurate_count": accurate_count,
        "bad_responses": bad_responses,
        "errors": errors,
        "time_taken": total_time
    }

def get_user_input():
    dataset_name = input("Enter the dataset name (e.g., math, gpqa, mmlu, mmlu-pro): ").strip().lower()
    subject = ""
    if dataset_name in ["mmlu", "mmlu-pro"]:
        subject = input(f"Enter the {dataset_name.upper()} subject: ").strip()
    
    model = input("Enter the model to evaluate (e.g., Qwen/Qwen2-72B-Instruct): ").strip() or "Qwen/Qwen2-72B-Instruct"
    evaluator_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    
    while True:
        try:
            num_questions = int(input("Enter the number of questions to evaluate (0 for entire dataset): "))
            if num_questions >= 0:
                break
            print("Please enter a non-negative number.")
        except ValueError:
            print("Please enter a valid number.")
    
    return dataset_name, subject, model, evaluator_model, num_questions

def load_dataset_questions(dataset_name: str, subject: str, num_questions: int, shuffle: bool = True) -> List[Dict[str, str]]:
    logs_folder = "logs"
    os.makedirs(logs_folder, exist_ok=True)
    print("THE DATASET NAME IS: ", dataset_name)
    formatted_file = os.path.join(logs_folder, f"formatted_{dataset_name}_{subject}_questions.json")
    
    if os.path.exists(formatted_file):
        print(f"Using existing formatted dataset: {formatted_file}")
        with open(formatted_file, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
    else:
        print(f"Loading and formatting new dataset: {dataset_name}")
        if dataset_name == "math":
            dataset = load_dataset("competition_math", split="test")
            full_dataset = [
                {
                    "instruction": item["problem"],
                    "output": item["solution"]
                }
                for item in dataset
            ]
        elif "gpqa" in dataset_name.lower():
            
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
            full_dataset = []
            for item in dataset:
                
                if item.get("Question"):
                    options = [
                        item["Pre-Revision Correct Answer"],
                        item["Pre-Revision Incorrect Answer 1"],
                        item["Pre-Revision Incorrect Answer 2"],
                        item["Pre-Revision Incorrect Answer 3"]
                    ]
                    
                    # Shuffle the options
                    random.shuffle(options)
                    
                    # Find the index of the correct answer after shuffling
                    correct_index = options.index(item["Pre-Revision Correct Answer"])
                    correct_answer = chr(65 + correct_index)  # Convert index to letter (A, B, C, D)
                    
                    instruction = f"{item['Question']}\n\nAnswer Choices:\n"
                    instruction += "\n".join([f"({chr(65+i)}) {option}" for i, option in enumerate(options)])
                    
                    full_dataset.append({
                        "instruction": instruction,
                        "output": f"The correct answer is {correct_answer}"
                    })
        elif dataset_name in ["mmlu", "mmlu-pro"]:
            if dataset_name == "mmlu":
                dataset = load_dataset("cais/mmlu", "all", split="test")
                full_dataset = [
                    {
                        "instruction": f"Subject: {item['subject']}\n\n{item['question']}\n\nAnswer Choices:\n" + "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(item["choices"])]),
                        "output": f"The correct answer is {chr(65 + item['answer'])}"
                    }
                    for item in dataset
                ]
            else:  # mmlu-pro
                dataset = load_dataset("TIGER-Lab/MMLU-Pro", subject, split="test")
                full_dataset = [
                    {
                        "instruction": item["question"] + "\n\nAnswer Choices:\n" + "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(item["options"])]),
                        "output": f"The correct answer is {item['answer']}"
                    }
                    for item in dataset
                ]

        with open(formatted_file, 'w', encoding='utf-8') as f:
            json.dump(full_dataset, f, indent=2)
    
    if num_questions > 0:
        return random.sample(full_dataset, min(num_questions, len(full_dataset)))
    if shuffle:
        return random.sample(full_dataset, len(full_dataset))  # Shuffle the entire dataset
    return full_dataset  # Return the entire dataset without shuffling

async def main():
    dataset_name, subject, model, evaluator_model, num_questions = get_user_input()
    
    # Load the dataset
    full_dataset = load_dataset_questions(dataset_name, subject, num_questions)
    
    total_questions = len(full_dataset)
    print(f"Total questions to evaluate: {total_questions}")

    # Process in batches of 10
    batch_size = 5
    all_results = []
    total_accurate = 0
    total_bad_responses = 0
    total_errors = 0
    total_time = 0

    for i in range(0, total_questions, batch_size):
        batch = full_dataset[i:i+batch_size]
        batch_results = await evaluate_model_batch(dataset_name, model, evaluator_model, batch)
        
        all_results.extend(batch_results['results'])
        total_accurate += batch_results['accurate_count']
        total_bad_responses += batch_results['bad_responses']
        total_errors += batch_results['errors']
        total_time += batch_results['time_taken']

        print(f"Processed {min(i+batch_size, total_questions)}/{total_questions} questions")

    # Calculate final accuracy and confidence interval
    final_accuracy = (total_accurate / total_questions) * 100 if total_questions > 0 else 0
    
    # Calculate 95% confidence interval
    z_score = 1.96  # for 95% confidence
    margin_of_error = z_score * math.sqrt((final_accuracy * (100 - final_accuracy)) / total_questions)
    ci_lower = max(0, final_accuracy - margin_of_error)
    ci_upper = min(100, final_accuracy + margin_of_error)

    results = {
        "dataset": dataset_name,
        "model": model,
        "total_questions": total_questions,
        "accuracy": final_accuracy,
        "confidence_interval": (ci_lower, ci_upper),
        "bad_responses": total_bad_responses,
        "errors": total_errors,
        "time_taken": total_time,
        "detailed_results": all_results
    }
    
    print("\n=== Evaluation Results ===")
    print(f"Dataset: {results['dataset']}")
    print(f"Model: {results['model']}")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Accuracy: {results['accuracy']:.2f}% (95% CI: {results['confidence_interval'][0]:.2f}% - {results['confidence_interval'][1]:.2f}%)")
    print(f"Bad Responses: {results['bad_responses']}")
    print(f"Errors: {results['errors']}")
    print(f"Time Taken: {results['time_taken']:.2f} seconds")
    
    # Save detailed results to a JSON file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logs_folder = "logs"
    os.makedirs(logs_folder, exist_ok=True)
    output_file = os.path.join(logs_folder, f"eval_results_{dataset_name}_{model.split('/')[-1]}_{total_questions}q_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {output_file}")

    # Calculate category-wise accuracies
    category_results = {}
    for result in all_results:
        category = classify_text(result['question'])  # You'll need to import this function
        if category not in category_results:
            category_results[category] = {'correct': 0, 'total': 0}
        category_results[category]['total'] += 1
        if result['is_accurate']:
            category_results[category]['correct'] += 1

    print("\n=== Category-wise Accuracies ===")
    for category, data in category_results.items():
        accuracy = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
        print(f"{category}: {accuracy:.2f}% ({data['correct']}/{data['total']})")

if __name__ == "__main__":
    asyncio.run(main())