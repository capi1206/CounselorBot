from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from langfuse.decorators import observe
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SummaryGen:
    """Class that makes summaries of the conversations, 
    must be initialized with a model unless the default one is wanted. 
"""
    def __init__(self, model_name="t5-base"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
    @observe()
    def create_summary(self, in_text):
        inputs = self.tokenizer("summarize: " + in_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=100,  # Adjusted max length
                min_length=7,
                length_penalty=1.5,  # Adjusted length penalty
                num_beams=4,
        early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], 
                                        skip_special_tokens=True)
        return summary

class EmbeddingModel:
    def __init__(self, model_name='bert-base-nli-mean-tokens'):
        self.model = SentenceTransformer(model_name)

    def give_embedding(self, phrase):
        return self.model.encode(phrase)

    def give_similarity(self, vector1, vector2):
        v = vector1.reshape(1, -1)
        w = vector2.reshape(1, -1)
        return cosine_similarity(v, w)[0][0]    
    
class PromptGen:
    def __init__(self, embeding_model:EmbeddingModel):
        self.embedding_model=embeding_model
        self.dict_topics ={
            "Communication Skills" :{ "embedding" : self.embedding_model.give_embedding("Communication Skills"),
                                     "preprompt" : " A Communication Skills Coach, an expert in effective communication and conflict resolution, "

        },
        "Emotional Intelligence" :{ "embedding" : self.embedding_model.give_embedding("Emotional Intelligence"),
                                   "preprompt" :" An Emotional Intelligence Coach, an expert in understanding and managing emotions in yourself and others, "

        },
        "Conflict Resolution" :{ "embedding": self.embedding_model.give_embedding("Conflict Resolution"),
                                "preprompt":" A Conflict Resolution Coach, an expert in navigating and resolving interpersonal conflicts, "

        },
        "Active Listening" :{ "embedding": self.embedding_model.give_embedding("Active Listening"),
                             "preprompt": "An Active Listening Coach, an expert in fostering effective communication through the art of listening, "
},
        "Feedback Techniques" :{ "embedding": self.embedding_model.give_embedding("Feedback Techniques"),
                                "preprompt":"A Feedback Techniques Coach, an expert in delivering and receiving constructive feedback effectively, "       }
        }
        
    @observe()    
    def gen_prompt(self, query, context=""):
        query_embedding = self.embedding_model.give_embedding(query)
        topic=sorted(self.dict_topics.keys(),
               key=lambda x: self.embedding_model.give_similarity(      #this piece of code organizes the coach functionality with respect
                   query_embedding, self.dict_topics[x]["embedding"]    #to cosine similarity of the embedding with respect to
               ), reverse =True)[0]                                    #the query and returns the topic that matches the most.
        prompt =""
        if context:
            prompt += ". In previous conversations it was mentioned: " + context + ". "

        prompt += "The user comes with the following question: \"" +query +"\""
        #adds context to the query, if passed.

        prompt += self.dict_topics[topic]["preprompt"]+ "advises the user to \""
        return prompt

class LLMModel:
    def __init__(self, model_name ="EleutherAI/gpt-neo-1.3B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def _simplify_response(self, response):
    #index where the sentene "advises the user to" is found
        try:
            i=response.index("advises the user to \"")
        except:
            print("the generated response is in an unexpected format!")
            return None
        answer = response[i+len("advises the user to \""):]
        return answer.split('"')[0]
        
    @observe()
    def query_model(self, prompt):
        """Produces a response from the llm, ready to be given back to the user.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(inputs['input_ids'],
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # Prevent repeating 2-grams
            temperature=0.7,  # this coniguration helps avoiding repeated sentences as a response.
            top_k=50,
            top_p=0.95
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response =self._simplify_response(response)
        return response        
    
