import warnings
warnings.filterwarnings("ignore")


import os
import yaml
import pickle
import re
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
from openai import OpenAI


# Loading env variables
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')


# Config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)




class OptimizerLLM:
    def __init__(self):
        self.client = OpenAI()


    def generate(self, meta_prompt, num_candidates=4, temperature=1.0):   # num_candidates=8
        
        candidate_prompts = []
        
        for _ in range(num_candidates):
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a prompt optimization assistant."},
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=temperature
            )
            
            generated_prompt = response.choices[0].message.content
            candidate_prompts.append(generated_prompt)
        
        return candidate_prompts


class ScorerLLM:
    def __init__(self):
        self.client = OpenAI()
    
    def generate(self, prompt, input_text, temperature=0.0):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content




class opro_class():

    def __init__(self):
        pass
    

    def castint(self, input_int):
        try:
            input_int = re.findall(r"\d+", input_int)[0]
            casted_int=int(input_int.strip())
        except:
            print(input_int)
            casted_int=-1
        return casted_int


    def load_data(self, config):
        # Loading dataset
        dataset_path = config['file_paths']['hf_data_subset']
        with open(dataset_path, r"rb") as f:
            data = pickle.load(f)

        questions, answers = map(list, zip(*[(d['question'], d['answer']) for d in data]))

        answer_values = [self.castint(re.split(r"####", ans)[-1]) for ans in answers]

        return questions, answers, answer_values
    

    def chat_completion(self, prompt, temperature=0.2):
        client = OpenAI()
        return client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature
        ).choices[0].message.content
    


    def create_initial_meta_prompt(self, task, examples):
        meta_prompt = f"""
        Task Description:
        {task}

        Your goal is to generate an effective GENERAL-PURPOSE prompt that will help the model solve ANY math word problem, not just specific examples.

        The prompt should:
        1. Guide the model to solve ANY math word problem step by step
        2. Be general enough to work with different types of math problems
        3. Not contain specific numbers or problem details
        4. Focus on the problem-solving approach

        PREVIOUS_BEST_PROMPTS:
        []

        TRAINING_EXAMPLES:"""
        
        for idx, (input_text, target) in enumerate(examples, 1):
            meta_prompt += f"""
            Example {idx}:
            Input: {input_text}
            Expected Output: {target}"""
        
        meta_prompt += """
        
        Based on the task description and training examples above, generate a new prompt that will:
        1. Improve upon previous attempts (if any)
        2. Help the model generate accurate responses
        3. Be clear and specific
        4. Report the final answer with 4 hashtags before it. For example if the answer is 1000, at the end of your response print: #### 1000 

        Generate your prompt now:"""
        
        return meta_prompt
    

    def evaluate_prompt(self, prompt, training_examples, scorer_llm, temperature=0.0):

        correct = 0
        total = len(training_examples)
        
        for input_text, target in training_examples:
            response = scorer_llm.generate(
                prompt=prompt,
                input_text = input_text,
                temperature=temperature
            )
            
            response = self.castint(response.split(r"####")[-1].strip())

            # Compare response with target
            if response == target:
                correct += 1
        
        accuracy = correct / total
        return accuracy




    def update_meta_prompt(self, meta_prompt, best_prompts):

        # Format the best prompts and their accuracies
        best_prompts_formatted = "[\n"
        for prompt, accuracy in best_prompts:
            # Format each prompt-accuracy pair
            best_prompts_formatted += f'    ("{prompt}", {accuracy:.3f}),\n'
        best_prompts_formatted += "]"



        before_prompts, after_prompts = meta_prompt.split("PREVIOUS_BEST_PROMPTS:")
        _, after_examples = after_prompts.split("TRAINING_EXAMPLES:")


        # Reconstruct the meta prompt
        updated_meta_prompt = (
            f"{before_prompts}"
            f"PREVIOUS_BEST_PROMPTS:\n{best_prompts_formatted}\n\n"
            f"TRAINING_EXAMPLES:{after_examples}"
        )
        
        return updated_meta_prompt



    def get_best_prompt(self, best_prompts):

        if not best_prompts:
            raise ValueError("No prompts available to evaluate")
        
        
        sorted_prompts = sorted(best_prompts, key=lambda x: x[1], reverse=True)
        best_prompt, best_accuracy = sorted_prompts[0]
        
        print(f"Best prompt found with accuracy: {best_accuracy:.3f}")
        print(f"Prompt: {best_prompt}")
        
        return best_prompt, best_accuracy


    def opro_optimization(self, task_description, training_examples, max_iterations = 10): # 50
        
        optimizer_llm = OptimizerLLM()
        scorer_llm = ScorerLLM()

        # Create initial meta prompt with the training examples
        meta_prompt = self.create_initial_meta_prompt(
            task=task_description,
            examples=training_examples
    )
        

        best_prompts = []

        for iteration in tqdm(range(max_iterations), desc = r"OPRO Iteration"):
            # Generate candidate prompts using Optimizer LLM
            candidate_prompts = optimizer_llm.generate(meta_prompt)


                # Evaluate each candidate prompt using the same training examples
            for prompt in candidate_prompts:
                accuracy = self.evaluate_prompt(
                    prompt, 
                    training_examples,
                    scorer_llm,
                    temperature=0.0
                )
                best_prompts.append((prompt, accuracy))
            
            # Update meta-prompt with top performing prompts
            best_prompts = sorted(best_prompts, key = lambda x: x[1], reverse = True)[:20]
            meta_prompt = self.update_meta_prompt(meta_prompt, best_prompts)
        
        return self.get_best_prompt(best_prompts)




task_description = """Create a general-purpose prompt that guides the model to solve any math word problem.
The prompt should work for different types of math problems including addition, subtraction, multiplication, and division.
The prompt should encourage step-by-step problem solving without being specific to any particular problem."""



opro_training_set_path = config['file_paths']['opro_training_subset']
with open(opro_training_set_path, r"rb") as f:
    examples = pickle.load(f)





def main():
    opro = opro_class()

    best_prompt, best_accuracy = opro.opro_optimization(
        task_description=task_description,
        training_examples=examples
    )


if __name__ == "__main__":
    main()