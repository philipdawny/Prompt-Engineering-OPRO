import os
import re
import pickle
import random
import yaml
from datasets import load_dataset



# Config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)



def store_hf_dataset(data, name, save_path):

    name = re.split(r'/', name)[-1]

    if not os.path.exists(save_path):
        print(r">>> Path not found. Saving to parent directory. ")
        save_path = os.path.dirname(os.getcwd())
    
    
    save_path = os.path.join(save_path, f"{name}.pkl" )
    
    with open(save_path, 'wb') as f1:
        pickle.dump(data, f1)

    print(f"Dataset Saved: {save_path}")




def load_hf_dataset(dataset_name, save_path):

    try:
        data = load_dataset(dataset_name, 'main')
        store_hf_dataset(data, dataset_name, save_path)
            
    except:
        print("HF dataset not found!")

    
def main():

    hf_data_path = config['hf_data']['hf_dataset_name']
    save_path = config['hf_data']['file_path_to_save']
    sample_size = config['hf_data']['random_subset_size']

    load_hf_dataset(hf_data_path, save_path)

if __name__ == "__main__":
    main()
    

    