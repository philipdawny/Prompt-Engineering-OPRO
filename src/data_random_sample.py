import os
import yaml
import re
import pickle
import random

# Config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)



dataset_path = config['file_paths']['hf_data']
dataset_name = config['hf_data']['hf_dataset_name']
name = re.split(r'/', dataset_name)[-1]
random_sample_size = config['hf_data']['random_subset_size']
path_to_save = os.path.join(os.path.split(dataset_path)[0], f"{name}_{random_sample_size}_sample.pkl")



def sample():

    # Loading dataset
    with open(dataset_path, r"rb") as f:
        data = pickle.load(f)

    # Generating random sample
    data = random.sample(list(data['train']), random_sample_size)

    with open(path_to_save, r"wb") as f1:
        pickle.dump(data, f1)


def main():
    sample()


if __name__ == "__main__":
    main()