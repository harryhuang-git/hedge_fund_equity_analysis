import openai
import json
import os
from typing import List, Dict, Any

class LLMFactorGenerator:
    def __init__(self, openai_api_key=None, cache_file='data/llm_factors.json'):
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.cache_file = cache_file
        self.factors = self._load_cache()
        if self.api_key:
            openai.api_key = self.api_key

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.factors, f, ensure_ascii=False, indent=2)

    def prompt_factor_formula(self, structured_data: Dict[str, Any], prompt: str) -> str:
        # Use OpenAI GPT to generate a factor formula
        messages = [
            {"role": "system", "content": "You are a quantitative researcher. Suggest robust factor formulas for alpha discovery."},
            {"role": "user", "content": prompt + f"\nData: {structured_data}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=256
        )
        factor_formula = response['choices'][0]['message']['content'].strip()
        return factor_formula

    def generate_and_store_factor(self, structured_data: Dict[str, Any], prompt: str, factor_name: str):
        formula = self.prompt_factor_formula(structured_data, prompt)
        self.factors[factor_name] = {
            'prompt': prompt,
            'formula': formula,
            'data_example': structured_data
        }
        self._save_cache()
        return formula

    def summarize_sentiment(self, text: str) -> str:
        # Use LLM to summarize sentiment from news or Q&A
        messages = [
            {"role": "system", "content": "You are a financial news sentiment analyst. Summarize the sentiment (positive, negative, neutral) and key points."},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=128
        )
        return response['choices'][0]['message']['content'].strip()

    def classify_event(self, text: str) -> str:
        # Use LLM to classify breaking events or entity mentions
        messages = [
            {"role": "system", "content": "You are a financial event classifier. Classify the event type and affected stocks/entities."},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=128
        )
        return response['choices'][0]['message']['content'].strip()

    def get_all_factors(self) -> Dict[str, Any]:
        return self.factors 