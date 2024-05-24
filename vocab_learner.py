import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

import pandas as pd
from dotenv import load_dotenv
from IPython.display import Markdown
from llama_index.core.llms import ChatMessage
from llama_index.llms.llama_api import LlamaAPI
from llama_index.llms.ollama import Ollama

import logging


@lru_cache(maxsize = None)
def get_model():
    print("Initializing llama3 8B local model")
    return Ollama(model = 'llama3')

@dataclass
class VocabBook:
    english_words: List[str]
    chinese_words: List[str]

    def __post_init__(self):
        if not len(self.english_words) ==  len(self.chinese_words):
            raise ValueError("dictionary length inconsistent")
        self.n_words = len(self.english_words)

    def __getitem__(self, idx:int):
        return self.english_words[idx]

    def __len__(self):
        return len(self.english_words)
    
    @classmethod
    def from_excel(cls, path:str = 'words.xlsx'):
        df = pd.read_excel(path, header = None)
        df = df.iloc[:,2:]
        df.columns = ['english','chinese']
        english_words = df['english'].tolist()
        chinese_words = df['chinese'].tolist()
        return cls(english_words, chinese_words)

    def sample_english(self):
        return self.english_words[random.randint(0, self.n_words)]

    def sample_chinese(self):
        return self.chinese_words[random.randint(0, self.n_words)]


TUTORIAL_PROMPT = """
You are an English language teacher. Your task is to explain the meaning, part of speech, \
and various forms of the given word in a way that is easy for a beginner to understand. Follow this structure:

Provide the definition(s) of the word in simple language.
List possible parts of speech for the word.
For each part of speech:

If it's a noun, determine it is countable or non-count, if it is countable, provide the plural form.
If it's a verb, show the past tense, present participle, and past participle forms.
If it's an adjective, provide the comparative and superlative forms.

Give examples of common phrases or word groups that include the word.
Provide several sample sentences using the word in different contexts.
If the word is commonly used with certain prepositions, explain which prepositions are typically used with the word and in what contexts.

Remember to explain everything using straightforward language, avoiding complex terminology or jargon. Your explanations should be concise yet comprehensive.
The word is: {current_word}

MAKE SURE that your response is a nice formatted markdown document, with no markdown headers.
""".strip()


QA_SYSTEM_PROMPT = """
You are an English teacher. Your task is to responde to students' questions in detail given a specific English word.
"""

@lru_cache(maxsize = None)
def get_qa_messages(word):
    return [ChatMessage(role="system", content= QA_SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"The current word in questions is: {word}. "),
            ChatMessage(role = "assistant", content = "Got it.")]

@dataclass
class Learner:
    mode:str = 'random'
    start_index:int = 0
    vocab_book:VocabBook = field(default_factory = VocabBook.from_excel)
    model: Ollama = field(default_factory = get_model)
    jupyter:bool = True
    
    def learn_word(self, word:str):
        if word not in self.vocab_book.english_words:
            raise ValueError("word not found")

        self.curr_index = self.vocab_book.english_words.index(word)
        self.current_word = self.vocab_book[self.curr_index]
        self.reset_messages()
        
    def __post_init__(self):
        self.n_words = len(self.vocab_book)
        
        if self.mode == 'random':
            self.start_index = random.randint(0, self.n_words)

        elif self.mode == "sequential":
            pass
            
        else:
            raise ValueError(f"Undefined learning mode `{self.mode}`")

        self.curr_index = self.start_index

        self.current_word = self.vocab_book[self.curr_index]

        self.reset_messages()

    def reset_messages(self):
        self.messages = get_qa_messages(self.current_word)
        return self
    
    def get_word(self):
        if self.jupyter:
            return Markdown(f"Word: **{self.current_word}**")
        else:
            return f"*Word:* **{self.current_word}**"
            
    def next(self):
        if self.mode == "random":
            self.curr_index = random.randint(0, self.n_words)
        else:
            self.curr_index = (self.curr_index + 1) % self.n_words

        self.current_word = self.vocab_book[self.curr_index]

        self.reset_messages()
        
        if self.jupyter:
            return Markdown(f"Next word is: **{self.current_word}**")
        
        else:
            return f"*Word:* **{self.current_word}**"

    def explain(self):
        completion = self.model.complete(TUTORIAL_PROMPT.format(current_word = self.current_word))
        if self.jupyter:
            return Markdown(completion.text)
        else:
            return completion.text

    def ask(self, prompt):
        self.messages.append(ChatMessage(role="user", content=prompt))
        resp = self.model.chat(self.messages)
        self.messages.append(ChatMessage(role = 'assistant', content = resp.message.content))
        if self.jupyter:
            return Markdown(resp.message.content)
        else:
            return resp.message.content        
      
     
def get_learner(type = 'local',mode:str = 'random', llama_api_key = None, jupyter = True, remote_model_name:str = 'mixtral-8x7b-instruct'):
    
    if type == 'remote':
        # if no api key is provided, try to load from .env file
        if llama_api_key is None:
            load_dotenv()
            llama_api_key = os.getenv("LLAMA_API_KEY")
            if llama_api_key is None:
                raise ValueError("No Llama API key provided")

        print(f"Loading remote model {remote_model_name}")
        return Learner(
            mode = mode,
            model = LlamaAPI(
                api_key = llama_api_key,
                model = remote_model_name, 
                max_tokens = 1024, 
                temperature = 0, 
                ),
            jupyter = jupyter
            )

    return Learner(mode = mode, jupyter = jupyter)