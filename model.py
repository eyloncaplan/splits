from abc import ABC, abstractmethod
import os
import json
import hashlib

class Model(ABC):
    # All cache files will be stored in this directory.
    CACHE_DIR = os.path.join(os.environ.get("SCRATCH_PATH", "/default/scratch/path"), "prompt_caches")

    def __init__(self):
        self.cache = {}  # Individual prompt-level cache.
        # Load persistent cache if it exists.
        cache_file = self.get_cache_filename()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print("Error loading cache:", e)

    def get_cache_filename(self):
        """Compute a unique filename for the cache based on the model's parameters."""
        key_dict = self.get_cache_key()  # Each subclass must implement this.
        key_json = json.dumps(key_dict, sort_keys=True)
        hash_key = hashlib.md5(key_json.encode()).hexdigest()
        model_name = self.__class__.__name__
        filename = f"{model_name}_{hash_key}.json"
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        return os.path.join(self.CACHE_DIR, filename)

    def save_cache(self):
        """Saves the current cache to the persistent file."""
        cache_file = self.get_cache_filename()
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except Exception as e:
            print("Error saving cache:", e)

    def infer(self, data, **kwargs):
        """
        Runs inference on a dictionary containing:
          - 'prompts': a list of prompt strings
          - (optional) 'sys_prompt': a system prompt string
        Caches results at the individual prompt level.
        If a prompt (with its system prompt) has been seen before, its response is retrieved from the cache.
        Duplicate prompts in the same call are processed only once.
        """
        sys_prompt = data.get('sys_prompt', None)
        all_prompts = data.get('prompts', [])
        results = [None] * len(all_prompts)
        # Build a dictionary mapping unique keys to a list of indices.
        unique_cache = {}
        for i, prompt in enumerate(all_prompts):
            key = json.dumps({'prompt': prompt, 'sys_prompt': sys_prompt}, sort_keys=True)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                if key not in unique_cache:
                    unique_cache[key] = {'prompt': prompt, 'indices': [i]}
                else:
                    unique_cache[key]['indices'].append(i)
        
        # Process unique prompts that are not cached.
        unique_keys = list(unique_cache.keys())
        unique_prompts = [unique_cache[k]['prompt'] for k in unique_keys]
        new_results = []
        # Process these unique prompts in batches.
        for j in range(0, len(unique_prompts), self.batch_size):
            batch_prompts = unique_prompts[j:j+self.batch_size]
            batch_data = {'prompts': batch_prompts}
            if sys_prompt is not None:
                batch_data['sys_prompt'] = sys_prompt
            responses = self._infer(batch_data, **kwargs)
            new_results.extend(responses)
        
        # Update cache and results for all indices.
        for key, response in zip(unique_keys, new_results):
            for i in unique_cache[key]['indices']:
                results[i] = response
            self.cache[key] = response
        
        # Save cache to disk so it persists between runs.
        self.save_cache()
        return results

    @abstractmethod
    def _infer(self, data, **kwargs):
        """
        Runs inference on the given dictionary (uncached).
        Subclasses must implement this.
        """
        pass

    @abstractmethod
    def load_prompt(self, prompt_name, system_prompt=False):
        """
        Loads a prompt given its name. Subclasses implement custom behavior.
        """
        pass

    @abstractmethod
    def get_cache_key(self):
        """
        Returns a dictionary of parameters that uniquely identifies this model's configuration.
        Subclasses must implement this.
        """
        pass
