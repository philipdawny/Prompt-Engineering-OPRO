import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import random
import re
import yaml



# Config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Loading dataset
dataset_path = config['file_paths']['hf_data']


def castint(input_int):
    try:
        input_int = re.findall(r"\d+", input_int)[0]
        casted_int=int(input_int.strip())
    except:
        print(input_int)
        casted_int=-1
    return casted_int




def load_data(sample_size = 30):
    
    global dataset_path
    
    with open(dataset_path, r"rb") as f:
        data = pickle.load(f)




    data = random.sample(list(data['test']), sample_size)

    data = [tuple(i.values()) for i in data]

    data = [(i[0], castint(re.split(r"####", i[1])[-1])) for i in data]

    path_to_save = os.path.join(os.path.split(dataset_path)[0], f"gsm8k_opro_{sample_size}_sample.pkl")
    
    with open(path_to_save, r"wb") as f:
        pickle.dump(data, f)


def main():
    load_data()


if __name__ == "__main__":
    main()