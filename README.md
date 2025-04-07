# Prompt Engineering Techniques and Optimization by PROmpting

This repo contains for prompt engineering on the openai/gsm8k math word problems dataset.

##  Prompt Engineering Approaches:
  1. Propmting with the question.
  2. Basic PE with role setting and simple instructions defining appraoch and required output format.
  3. Few-Shot Prompting - Enhacing the prompt with a few example question-answer pairs for a few-shot learning approach.
  4. OPRO - Automated prompt augmentation using [OPRO](https://arxiv.org/abs/2309.03409#:~:text=In%20this%20work%2C%20we%20propose,is%20described%20in%20natural%20language.) - Optimization by PROmpting, developed by Google DeepMind.


## Steps to run code:

1. Run requirements.txt:

       !pip install requirements.txt
   
2. Update environment variables in ```.env```:

   * OpenAI API key
          

3. Update config variables in ```config.yaml```:

    hf_data
    *  ```path_to_save``` - Path to download the HuggingFace dataset
    * ```random_subset_size``` - Size of random sample from the dataset

    file_paths
    * ```hf_data``` - Path to the dataset pickle file
    * ```hf_data_subset``` - Path to the pickle file with sampled data

    
4. Run ```load_hf_data.py```
   
     This will download the [gsm8k](https://huggingface.co/datasets/openai/gsm8k) dataset and store it in  ```path_to_save```


5. Run ```data_random_sample.py```


     A subset of 250 question-answer pairs will also be saved in the same directory. This is used for evaluation of prompting techniques.  




7. Run ```scoring.py```  - Generates accuracy of GPT-3.5 Turbo answers without any prompt engineering  


8. Run ```scoring_new_prompts.py```  - Generates accuracy of GPT-3.5 Turbo answers with basic prompt engineering and Few-Shot prompting  


9. Run ```generate_opro_training_examples.py```  - Generates a set of 30 training question-answer examples for OPRO iterative prompt optimization.  



10. Run ```opro.py```  - Run OPRO optimization  



11. Run ```opro_scoring.py```  
