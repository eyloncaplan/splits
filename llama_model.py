from model import Model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class LlamaModel(Model):
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", 
                 max_new_tokens=400, temperature=0.001, top_p=1,
                 batch_size=8, **kwargs):
        """
        Initializes the Llama model wrapper.
        
        Parameters:
            model_name (str): The Hugging Face model identifier.
            max_new_tokens (int): Maximum new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p (nucleus sampling) value.
            batch_size (int): How many prompts to process in a single batch.
            **kwargs: Additional parameters if needed.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size

        self.prompt_separated = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # ).to(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        super().__init__()

    def _infer(self, data, **kwargs):
        """
        Runs inference on a dictionary containing prompts in batches.
        
        The dictionary should have:
            - 'prompts': a list of prompt strings.
        
        Returns:
            A list of generated responses.
        """
        prompts = data.get('prompts', [])
        responses = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i: i + self.batch_size]
            # inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to('cuda')
            # Get the fixed sequence length in the batch (since left-padding is used)
            prompt_length = inputs["input_ids"].size(1)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True
                )
            
            for j in range(outputs.size(0)):
                generated_tokens = outputs[j][prompt_length:]
                responses.append(self.tokenizer.decode(generated_tokens, skip_special_tokens=True))
        return responses

    def load_prompt(self, prompt_name):
        # For Llama, system/user components are baked into one prompt.
        directory = os.path.join("prompts", "llama_prompts")
        filename = f"{prompt_name}.txt"
        
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    def get_cache_key(self):
        return {
            'model_name': self.model_name,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p
        }
