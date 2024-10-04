import os


from transformers import T5ForConditionalGeneration, T5Tokenizer

from vectordb import PineConeDB

class SummaryGen:
    def __init__(self, model_name="t5-base"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def create_summary(self, in_text):
        inputs = self.tokenizer("summarize: " + in_text, return_tensors="pt", max_length=1000, truncation=True)
        summary_ids = self.model.generate(inputs["input_ids"], max_length=200, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

sum_gen=SummaryGen()
sum_gen.create_summary("Hey pal,what is up? \n not you much, you know the same thing, other than this and that... \n what is up with you? \n did you know that I got married about a month ago? It was a while since we did not catch up")
#####----------------------------------------------------------------------------------------------------------

class PineConeDB:
    def __init__(self, index_name, dimension):
        self.pc = Pinecone(api_key="9be85a06-9767-4b98-8a3f-7a94b2d1cffd")
        self.index_name = index_name
        self.dimension = dimension  
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                    )
                )
            print("Index created successfully.")
        else:
            print("Index already exists.")

    def save_summary(self, summary, summary_embedding):
        # Save the summary and its embedding in the vector database
        # Each summary should have a unique identifier, here we use a simple id based on the length of the index
        id = f"summary-{len(self.pc.index(self.index_name).fetch()) + 1}"
        
        self.pc.index(self.index_name).upsert(
            vectors=[(id, summary_embedding, {'summary': summary})]
        )
        print(f"Summary saved with ID: {id}")

    def give_relevant_summary(self, query_embedding, top_k=1):
        # Perform similarity search in the vector database
        response = self.pc.index(self.index_name).query(
            top_k=top_k,
            vector=query_embedding,
            include_metadata=True
        )
        
        if response and response['matches']:
            return [match['metadata']['summary'] for match in response['matches']]
        else:
            return []        

    

##create an embedding for a particular phrase------------------------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the model model_name ="bert-nli-mean-tokens"
class EmbeddingModel:
    def __init__(self, model_name='bert-base-nli-mean-tokens'):
        self.model = SentenceTransformer(model_name)
    
    def give_embedding(self, phrase):
        return self.model.encode(phrase)

    def give_similarity(self, vector1, vector2):
        v = vector1.reshape(1, -1)
        w = vector2.reshape(1, -1)
        return cosine_similarity(v, w)[0][0]

###--------------------------------------------------------------------------------------
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
    def gen_prompt(self, query, context=""):
        query_embedding = self.embedding_model.give_embedding(query)
        topic=sorted(self.dict_topics.keys(), 
               key=lambda x: self.embedding_model.give_similarity(      #this piece of code organizes the items with respect
                   query_embedding, self.dict_topics[x]["embedding"]    #to cosine similarity of the embedding with respect to
               ), reverse =True)[0]                                    #the query and returns the topic that matches the most.
        prompt = "An user comes with the following question: \"" +query +"\""
        #adds context to the query, if passed.
        if context:
            prompt += " in previous conversations it was mentioned: " + context
        prompt += self.dict_topics[topic]["preprompt"]+ "advises the user to "

        return prompt   

####-------------------------------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMModel:
    def __init__(self, model_name ="EleutherAI/gpt-neo-1.3B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def _simplify_response(self, response):
    #index where the sentene "advises the user to" is found
        try:
            i=response.index("advises the user to")
        except:
            print("the generated response is in an unexpected format!")
            return None
        answer = response[i+len("advises the user to"):]
        return answer
    
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
        return self._simplify_response(response)
    
# Example usage

emb_mod =EmbeddingModel()
prompt_e =PromptGen(emb_mod)
prompt = prompt_e.gen_prompt("I have a few issues with my superior.")
res=query_model(prompt)
print(res)
response = query_model(prompt)
print(response)

emb_mod =EmbeddingModel()
prompt_e =PromptGen(emb_mod)
prompt = prompt_e.gen_prompt("I have issues receiving feedback form my superiors at work.")
prompt
# Tokenize input
inputs = tokenizer(prompt, return_tensors='pt')

# Generate a response
outputs = model.generate(inputs['input_ids'], max_length=215)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)


##query a summary of a relevant conversation given a particular query

##create a summary of a conversation---------------------------------------------------------------------------------------------------

def collect_user_input():
    session_conversation = """I am an AI chatbot coach for soft skills, begin by asking me something.
        I answer standalone questions. To finish the conversation  type 'exit'."""
    print(session_conversation) #at this point the conversation only goes with the welcoming message
    ##Initialize the PineConeDB, SummaryGen, EmbeddingModel, and the TODO PromptGen, LLMModel
    #user_if =get userid
    embedding_model=EmbeddingModel()
    embedding_dimension=embedding_model.give_embedding("sample").shape[0]
    vector_db = PineConeDB(user_id, embedding_dimension)
    summary_gen = SummaryGen()
    prompt_gen = PromptGen() ##may just be a function
    

    while True:
        
        user_input = input()  # Get the question from the user

        if user_input.lower() == 'exit':
            break  # Exit the loop if the user types 'exit'
        
        session_conversation += "\n + User: " + user_input  + '\n'# Save the input in the list

        ##generate a search in the vdb to find the right context
        #context = summary of the most relevant conversation closest to the query

        ##generate a prompt with the previous information
        #prompt = a generated prompt that depends on the user's query and the given context from the vectordb

        ##ask the LLM with the given prompt. Get an answer back
        # answer = what the LLM gives out of the prompt
        #session_conversation += "Bot: " + answer  + '. I hope that answers your question, you can ask me something else ...'
        print('I hope that answers your question, you can ask me something else ...')
    # make a summary of the saved conversation
    conversation_summary = summary_gen.create_summary(session_conversation)   
    # take an embedding of that summary
    conversation_embedding = embedding_model.give_embedding(conversation_summary)
    # save the embedding of the summary and its summary in the vdb    
    vector_db.save_summary(conversation_summary, conversation_embedding)


    return session_conversation  # Return the list of user inputs

# Run the function
collected_texts = collect_user_input()

# Output the collected texts
print("\nCollected texts:", collected_texts)
for text in collected_texts:
    print(text)