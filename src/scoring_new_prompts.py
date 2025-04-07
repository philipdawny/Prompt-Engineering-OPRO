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


# OpenAI chat API
def chat_completion(prompt, temperature=0.2):
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



prompt_template = """You are a mathematician assistant who can perform medium level mathematical computations and provide the numerical answer.
                    You will be asked a math question based on a situation, you will need to perform the calculation based on the provided information and report the answer.

                    Steps to answer the question:
                    1. Process the given information from the input question.
                    2. Understand what is required to be answered
                    3. Reason for each step of the calculation
                    4. Report the final answer with 4 hashtags before it. For example if the answer is 1000, at the end of your response print: #### 1000 


                    Question: {user_question}  """



few_shot_prompt_template = """You are a mathematician assistant who can perform medium level mathematical computations and provide the numerical answer.
                    You will be asked a math question based on a situation, you will need to perform the calculation based on the provided information and report the answer.

                    Steps to answer the question:
                    1. Process the given information from the input question.
                    2. Understand what is required to be answered
                    3. Reason for each step of the calculation
                    4. Report the final answer with 4 hashtags before it. For example if the answer is 1000, at the end of your response print: #### 1000 


                    Refer to the below examples to understand the approach to solve the given problem:

                    Example 1:
                    Question: A couple with two children, ages 6 and 10 years old, decided to go to an amusement park. The regular ticket costs $109, but children below 12 years old have a $5 discount. If they gave the cashier $500, how much change will they receive?
                    Reasoning and Answer: The ticket costs 109 - 5 = 104 for each child.So the ticket cost of the two children is 104 x 2 = 208. The couple needs to pay the regular price, so it's 109 x 2 = 218. Thus, the family needs to pay a total of 208 + 218 = 426. Therefore, their change is 500 - 426 = 74. #### 74

                    Example 2:
                    Question: An Italian restaurant earns $600 every weekday and twice as much on the weekend. How much money does it earn by the end of the month?
                    Reasoning and Answer: On weekdays it earns $600/weekday * 5 weekdays = 3000. During the weekend it earns twice as much each day so it makes ($600 * 2)/weekend day * 2 weekend days = 2400. Every week it earns $3000 + $2400 = 5400. By the end of the month, it earns $5400/week * 4 weeks = $21600. #### 21600

                    Example 3:
                    Question: A phone tree is used to contact families and relatives of Ali's deceased coworker. Ali decided to call 3 families. Then each family calls 3 other families, and so on. How many families will be notified during the fourth round of calls?
                    Reasoning and Answer: In the first round, there are 3 families called. In the second round, there are 3 x 3 = 9 families called. In the third round, there are 9 x 3 = 27 families called. In the fourth round, there are 27 x 3 = 81 families called. #### 81

                    Example 4:
                    Question: Morgan's dad said that she had $90 budgeted for her birthday party. She wants to make sure she and her friends all get to play one round of mini-golf, have $5 in arcade tokens, and get to ride the go-karts twice. A round of mini-golf is $5. The Go-karts cost $10 a ride. How many friends can she invite?
                    Reasoning and Answer: The go karts will cost $20 per person because 10 x 2 = 20. Each person costs $30 because 5 + 5 + 20 = 30. Three total people can attend because 90 / 30 = 3. She can invite 2 friends because 3 - 1 = 2. #### 2

                    Example 5:
                    Question: Theo has $6000 he wishes to spend on his upcoming business trip to South Africa. He buys 6 business suits at $100 each, 3 suitcases at $50 each, a flight ticket that costs $700 more than 5 times as much as the cost of a business suit. He wishes to save $2000 for this trip, how much does he have to spend on buying gifts for his business partners in South Africa?
                    Reasoning and Answer: Theo buys business suits for 6 suits * $100/suit = $600. Theo buys suitcases for 3 suitcases * $50/suitcase = $150. Theo buys a flight ticket that costs $700 + 5 * $100 = $1200. Theo can spend $6000 - $600 - $150 - $1200 - $2000 = $2050 on gifts for his friends. #### 2050


                    Question: {user_question}  """



# GPT Answers
def generate_gpt_answers(questions, answer_values, answer_delimiter="####"):

    question_prompts = [prompt_template.format(user_question = question)  for question in questions]

    few_shot_question_prompts = [few_shot_prompt_template.format(user_question = question)  for question in questions]

    gpt_answers = [castint(chat_completion(q).split(answer_delimiter)[-1].strip()) for q in tqdm(question_prompts)]

    gpt_few_shot_answers = [castint(chat_completion(q).split(answer_delimiter)[-1].strip()) for q in tqdm(few_shot_question_prompts)]
    
    is_correct = [a == answer_values[i] for i,a in enumerate(gpt_answers)]
    num_wrong_format = np.sum(np.array(gpt_answers)==-1)

    few_shot_is_correct = [a == answer_values[i] for i,a in enumerate(gpt_few_shot_answers)]
    num_wrong_format_fewshot = np.sum(np.array(gpt_few_shot_answers)==-1)

    return np.mean(is_correct), num_wrong_format, gpt_answers, np.mean(few_shot_is_correct), num_wrong_format_fewshot, gpt_few_shot_answers


def main():
    accuracy, num_wrong_format, gpt_answers, few_shot_accuracy, few_shot_num_wrong_format, few_shot_gpt_answers = generate_gpt_answers(questions, answer_values)

    print(f""">>> GPT Accuracy: {np.round(accuracy, 2)*100}\n
          >>>"Wrong output format: {num_wrong_format}\n\n""")

    print(f""">>> GPT Accuracy with Few-Shot: {np.round(few_shot_accuracy, 2)*100}\n
          >>>"Wrong output format: {few_shot_num_wrong_format}""")


if __name__ == "__main__":
    main()


