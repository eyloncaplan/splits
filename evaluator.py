import json
import random
import pandas as pd
from llama_model import LlamaModel

MAX_CHARS = 2000

class Evaluator:
    def __init__(self, matching_df, batch_size=8, final_model=None, test=False):
        """
        Initializes the Evaluator with a matching DataFrame that includes a 'theory' column,
        and sets up a final classification model.
        
        Also adds a 'category' column to the DataFrame based on the 'description' column.
        
        Parameters:
            matching_df (pd.DataFrame): A DataFrame already augmented with predictions and a 'theory' column.
            batch_size (int): The batch size to use for inference.
            final_model (optional): An instance of a final classification model. If not provided,
                                    a default LlamaModel is created with meta-llama/Llama-3.1-70B-Instruct.
            test (bool): If True, no model is instantiated and evaluation will randomly select predictions.
        """
        self.matching_df = matching_df.copy()  # Work on a copy to avoid side effects
        self.batch_size = batch_size
        self.test = test

        # Load metadata/neutral_topics.json and add 'category' based on 'description'
        with open('metadata/neutral_topics.json', 'r') as f:
            neutral_topics = json.load(f)
        neutral_topic_to_category = {}
        for category, topics in neutral_topics.items():
            for topic in topics:
                neutral_topic_to_category[topic] = category
        self.matching_df['category'] = self.matching_df['description'].map(neutral_topic_to_category)
        
        if not self.test:
            if final_model is None:
                self.final_model = LlamaModel(
                    model_name="meta-llama/Llama-3.1-70B-Instruct", 
                    # model_name="meta-llama/Llama-3.2-1B-Instruct", 
                    max_new_tokens=400, 
                    temperature=0.001, 
                    top_p=1, 
                    batch_size=batch_size
                )
            else:
                self.final_model = final_model
            
            # Load the classification prompt template (assumes a unified prompt format)
            self.classify_prompt_template = self.final_model.load_prompt("classify_with_theory_and_demo")
        else:
            self.final_model = None
            self.classify_prompt_template = None

    def evaluate(self):
        """
        Evaluates predictions on the matching DataFrame.

        For each row, it constructs a final classification prompt by creating the following variables:
        - demo_a: taken from 'demographic_A'
        - demo_b: taken from 'demographic_B'
        - post_set1: constructed from the 'sample_A' column if available
        - post_set2: constructed from the 'sample_B' column if available
        - theories: processed from the 'theory' column

        The classification prompt is then created by evaluating the prompt template with these variables in scope.
        In test mode, predictions are randomly chosen ("A" or "B").
        The function adds two columns to the DataFrame:
        - 'prediction': The boolean prediction (False for "A", True for "B", or None on error)
        - 'correct': True if 'switched' equals the prediction, otherwise False

        Returns:
            pd.DataFrame: The updated matching DataFrame.
        """
        if self.test:
            predictions = []
            for _ in self.matching_df.iterrows():
                r = random.choice(["A", "B"])
                predictions.append(False if r == "A" else True)
        else:
            classification_prompts = []
            indices = []
            # Construct prompts using variables required by the prompt template.
            for idx, row in self.matching_df.iterrows():
                # Extract demo variables.
                demo_a = row['demographic_A']
                demo_b = row['demographic_B']
                
                # Construct post_set1 from 'sample_A' if available.
                if 'sample_A' in row and row['sample_A'] is not None:
                    sample_a = row['sample_A']
                    if isinstance(sample_a, list):
                        post_set1 = "\n" + "\n\n".join(
                            ['"""' + str(post).replace('\n', '\t')[:MAX_CHARS] + '"""' for post in sample_a]
                        )
                    else:
                        post_set1 = str(sample_a)
                else:
                    post_set1 = ""
                
                # Construct post_set2 from 'sample_B' if available.
                if 'sample_B' in row and row['sample_B'] is not None:
                    sample_b = row['sample_B']
                    if isinstance(sample_b, list):
                        post_set2 = "\n" + "\n\n".join(
                            ['"""' + str(post).replace('\n', '\t')[:MAX_CHARS] + '"""' for post in sample_b]
                        )
                    else:
                        post_set2 = str(sample_b)
                else:
                    post_set2 = ""
                
                # Process theories from the 'theory' column.
                theories = row['theory']
                theories = "\n".join([line.strip() for line in str(theories).splitlines() if line.strip()])
                
                # Create a local variable dictionary for eval.
                local_vars = {
                    'demo_a': demo_a,
                    'demo_b': demo_b,
                    'post_set1': post_set1,
                    'post_set2': post_set2,
                    'theories': theories,
                    'MAX_CHARS': MAX_CHARS
                }
                
                # Evaluate the classification prompt template with the local variables.
                prompt = eval(self.classify_prompt_template, {}, local_vars)
                classification_prompts.append(prompt)
                indices.append(idx)
            
            # Run the final model in batch.
            if classification_prompts:
                final_data = {'prompts': classification_prompts}
                final_responses = self.final_model.infer(final_data)
            else:
                final_responses = []
            
            predictions = []
            # Parse final responses to extract predictions.
            for i, _ in enumerate(indices):
                final_response = final_responses[i]
                try:
                    set1 = final_response.split("Post Set 1: ")[1].split("\n")[0].strip()
                    set2 = final_response.split("Post Set 2: ")[1].strip()
                    set1 = set1.strip('<').strip('>').strip('Group').strip()
                    set2 = set2.strip('<').strip('>').strip('Group').strip()
                except Exception:
                    predictions.append(None)
                    continue
                
                if set1 == set2:
                    predictions.append(None)
                elif set1 == "A":
                    predictions.append(False)
                elif set1 == "B":
                    predictions.append(True)
                else:
                    predictions.append(None)
        
        # Add predictions and correctness to the DataFrame.
        self.matching_df['prediction'] = predictions
        self.matching_df['correct'] = self.matching_df['switched'] == self.matching_df['prediction']
        return self.matching_df

    def split_by_demographic_A_demographic_B(self):
        """
        Groups the DataFrame by the unique pairs (demographic_A, demographic_B) and computes count and accuracy.
        
        Returns:
            pd.DataFrame: A DataFrame with columns 'demographic_A', 'demographic_B', 'count', and 'accuracy'.
        """
        df = self.matching_df.groupby(['demographic_A', 'demographic_B']).agg(
            count=('correct', 'size'),
            accuracy=('correct', 'mean')
        ).reset_index()
        return df[['demographic_A', 'demographic_B', 'count', 'accuracy']]

    def split_by_unique_demographic(self):
        """
        Groups the DataFrame by unique demographic values, where a row is included if either 
        demographic_A or demographic_B equals the unique demographic.
        
        Returns:
            pd.DataFrame: A DataFrame with columns 'unique_demographic', 'count', and 'accuracy'.
        """
        unique_demographics = set(self.matching_df['demographic_A'].unique()) | set(self.matching_df['demographic_B'].unique())
        data_list = []
        for d in unique_demographics:
            sub = self.matching_df[(self.matching_df['demographic_A'] == d) | (self.matching_df['demographic_B'] == d)]
            data_list.append({
                'unique_demographic': d,
                'count': len(sub),
                'accuracy': sub['correct'].mean() if len(sub) > 0 else None
            })
        return pd.DataFrame(data_list)

    def split_by_category(self):
        """
        Groups the DataFrame by the 'category' column and computes count and accuracy.
        
        Returns:
            pd.DataFrame: A DataFrame with columns 'category', 'count', and 'accuracy'.
        """
        df = self.matching_df.groupby('category').agg(
            count=('correct', 'size'),
            accuracy=('correct', 'mean')
        ).reset_index()
        return df[['category', 'count', 'accuracy']]

    def split_by_category_demographic_A_demographic_B(self):
        """
        Groups the DataFrame by (category, demographic_A, demographic_B) and computes count and accuracy.
        
        Returns:
            pd.DataFrame: A DataFrame with columns 'category', 'demographic_A', 'demographic_B', 'count', and 'accuracy'.
        """
        df = self.matching_df.groupby(['category', 'demographic_A', 'demographic_B']).agg(
            count=('correct', 'size'),
            accuracy=('correct', 'mean')
        ).reset_index()
        return df[['category', 'demographic_A', 'demographic_B', 'count', 'accuracy']]

    def split_by_category_unique_demographic(self):
        """
        Splits the DataFrame by category and unique demographic. For each category and each unique demographic,
        includes rows where the 'category' matches and either demographic_A or demographic_B equals the unique demographic.
        
        Returns:
            pd.DataFrame: A DataFrame with columns 'category', 'unique_demographic', 'count', and 'accuracy'.
        """
        unique_demographics = set(self.matching_df['demographic_A'].unique()) | set(self.matching_df['demographic_B'].unique())
        data_list = []
        categories = self.matching_df['category'].dropna().unique()
        for cat in categories:
            for d in unique_demographics:
                sub = self.matching_df[(self.matching_df['category'] == cat) &
                                       ((self.matching_df['demographic_A'] == d) | (self.matching_df['demographic_B'] == d))]
                data_list.append({
                    'category': cat,
                    'unique_demographic': d,
                    'count': len(sub),
                    'accuracy': sub['correct'].mean() if len(sub) > 0 else None
                })
        return pd.DataFrame(data_list)

    def split_by_description_unique_demographic(self):
        """
        Splits the DataFrame by description and unique demographic. For each description and each unique demographic,
        includes rows where the description matches and either demographic_A or demographic_B equals the unique demographic.
        
        Returns:
            pd.DataFrame: A DataFrame with columns 'description', 'unique_demographic', 'count', and 'accuracy'.
        """
        unique_demographics = set(self.matching_df['demographic_A'].unique()) | set(self.matching_df['demographic_B'].unique())
        data_list = []
        descriptions = self.matching_df['description'].dropna().unique()
        for desc in descriptions:
            for d in unique_demographics:
                sub = self.matching_df[(self.matching_df['description'] == desc) &
                                       ((self.matching_df['demographic_A'] == d) | (self.matching_df['demographic_B'] == d))]
                data_list.append({
                    'description': desc,
                    'unique_demographic': d,
                    'count': len(sub),
                    'accuracy': sub['correct'].mean() if len(sub) > 0 else None
                })
        return pd.DataFrame(data_list)

    def split_by_description_demographic_A_demographic_B(self):
        """
        Groups the DataFrame by (description, demographic_A, demographic_B) and computes count and accuracy.
        
        Returns:
            pd.DataFrame: A DataFrame with columns 'description', 'demographic_A', 'demographic_B', 'count', and 'accuracy'.
        """
        df = self.matching_df.groupby(['description', 'demographic_A', 'demographic_B']).agg(
            count=('correct', 'size'),
            accuracy=('correct', 'mean')
        ).reset_index()
        return df[['description', 'demographic_A', 'demographic_B', 'count', 'accuracy']]

    def get_all_split_reports(self):
        """
        Returns a dictionary mapping each explanation of a split to the corresponding DataFrame.
        
        Splits included:
            - "Split by (demographic_A, demographic_B)"
            - "Split by (unique demographic)"
            - "Split by (category)"
            - "Split by (category, demographic_A, demographic_B)"
            - "Split by (category, unique demographic)"
            - "Split by (description, unique demographic)"
            - "Split by (description, demographic_A, demographic_B)"
        
        Returns:
            dict: A dictionary where keys are explanation strings and values are the DataFrames for each split.
        """
        splits = {
            "Split by (demographic_A, demographic_B)": self.split_by_demographic_A_demographic_B(),
            "Split by (unique demographic)": self.split_by_unique_demographic(),
            "Split by (category)": self.split_by_category(),
            "Split by (category, demographic_A, demographic_B)": self.split_by_category_demographic_A_demographic_B(),
            "Split by (category, unique demographic)": self.split_by_category_unique_demographic(),
            "Split by (description, unique demographic)": self.split_by_description_unique_demographic(),
            "Split by (description, demographic_A, demographic_B)": self.split_by_description_demographic_A_demographic_B()
        }
        return splits

    def get_accuracy(self):
        """
        Computes the overall accuracy of the predictions in the matching DataFrame.
        
        Returns:
            float: The overall accuracy.
        """
        return self.matching_df['correct'].mean()
