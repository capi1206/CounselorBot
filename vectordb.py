from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import random
import os
from langfuse.decorators import observe

#creates a random id for the vectordatabase, one per session
def gen_random_id():
    return ''.join([ chr(random.randint(97,122)) for _ in range(8)])

class PineConeDB:
    def __init__(self, dimension):
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index_name = gen_random_id()
        self.dimension = dimension
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
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
        self.index = self.pc.Index(self.index_name)

    def save_summary(self, summary, summary_embedding):
        """Insterts a summary of a conversation to be added as context,
         embedding of the summary is necessary for comparison.
         """
        id = f"summary-{self.index.describe_index_stats()['total_vector_count'] + 1}"

        self.index.upsert(
            vectors=[(id, summary_embedding, {'summary': summary})]
        )

    @observe()
    def give_relevant_summary(self, query_embedding, top_k=1):
        """gives the summary of the most relevant previous conversation
        with respect to the query.
        """
        response = self.index.query(
            top_k=top_k,
            vector=query_embedding,
            include_metadata=True
        )

        if response and response['matches']:
            return [match['metadata']['summary'] for match in response['matches']]
        else:
            return []