import warnings
warnings.filterwarnings("ignore")


import os
import yaml
import pickle
import re
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


# Loading dataset
dataset_path = config['file_paths']['hf_data_subset']
with open(dataset_path, r"rb") as f:
    data = pickle.load(f)

questions, answers = map(list, zip(*[(d['question'], d['answer']) for d in data]))


# Convert STR to INT
def castint(input_int):
    try:
        input_int = re.findall(r"\d+", input_int)[0]
        casted_int=int(input_int.strip())
    except:
        print(input_int)
        casted_int=-1
    return casted_int

answer_values = [castint(re.split(r"####", ans)[-1]) for ans in answers]



opro_final_prompt = """To effectively conquer any math word problem and derive the correct solution, adhere to this structured step-by-step strategy:

1. Begin by thoroughly understanding the problem statement to grasp the given information and identify the unknown quantity awaiting calculation.
2. Break down the problem into smaller, manageable components based on the required mathematical operations (addition, subtraction, multiplication, division).
3. Determine the suitable mathematical operation needed for each part based on the context provided in the problem scenario.
4. Formulate precise equations or expressions to accurately represent the relationships between the quantities involved.
5. Progressively solve each component step by step, ensuring accurate calculations are performed at every stage.
6. Verify your solution by checking that it satisfies all conditions specified within the problem statement.
7. Present your final answer with confidence and clarity, followed by 4 hashtags for easy identification.

Follow these structured steps diligently to navigate through diverse math word problems successfully and reach the correct solution. Let's now proceed to solve the math word problem step by step.
"""

# OpenAI chat API
def chat_completion(prompt, temperature=0.2):

    global opro_final_prompt
    client = OpenAI()
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": opro_final_prompt},
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    ).choices[0].message.content


# GPT Answers
def generate_gpt_answers(questions, answer_values, answer_delimiter="####"):

    question_prompts = [f"{question}. Output the final answer at the end with the prefix ####. For example: #### ANSWER" for question in questions]
    gpt_answers = [castint(chat_completion(q).split(answer_delimiter)[-1].strip()) for q in tqdm(question_prompts)]

    is_correct = [a == answer_values[i] for i,a in enumerate(gpt_answers)]
    num_wrong_format = np.sum(np.array(gpt_answers)==-1)

    return np.mean(is_correct), num_wrong_format, gpt_answers



def main():
    accuracy, num_wrong_format, gpt_answers = generate_gpt_answers(questions, answer_values)

    print(f""">>> GPT Accuracy: {np.round(accuracy, 2)*100}\n
          >>>"Wrong output format: {num_wrong_format}, {np.round(num_wrong_format/len(answer_values))*100}""")


if __name__ == "__main__":
    main()