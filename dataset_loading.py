# import time
# from datasets import load_dataset

# start = time.time()
# print("Loading dataset...")
# # dataset = load_dataset("ecaplan/splits", "all_seed_user_posts")
# dataset = load_dataset("ecaplan/splits", "all_seed_user_posts")
# print(f"Dataset features: {dataset['train'].features}")
# # print the first instance
# print("First instance:", dataset['train'][0])
# end = time.time()
# print(f"Dataset loaded in {end - start:.2f} seconds")





from datasets import load_dataset

all_seed_user_posts = load_dataset("ecaplan/splits", "all_seed_user_posts")['train']
high_groupness_user_posts = load_dataset("ecaplan/splits", "high_groupness_user_posts")['train']
high_groupness_by_topic = load_dataset("ecaplan/splits", "high_groupness_by_topic")['train']
splits = load_dataset("ecaplan/splits", "splits")['train']

# print the first instance
print(f"Dataset features: {dataset['train'].features}")
print("First instance:", dataset['train'][0])