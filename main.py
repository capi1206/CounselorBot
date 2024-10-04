from slack_bolt import App
from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode import SocketModeHandler
import certifi
import os

from assets import EmbeddingModel, SummaryGen, PromptGen, LLMModel
from vectordb import PineConeDB

load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()

SLACK_TOKEN = os.environ["SLACK_TOKEN"]
SIGNING_SECRET = os.environ["SIGNING_SECRET"]
SLACK_CHANNEL_ID = os.environ["SLACK_CHANNEL_ID"]
SLACK_APP_LEVEL_TOKEN = os.environ["SLACK_APP_LEVEL_TOKEN"]

app = App(token=SLACK_TOKEN, signing_secret=SIGNING_SECRET)

# Initialize the models and other resources
embedding_model = EmbeddingModel()
embedding_dimension = embedding_model.give_embedding("sample").shape[0]
vector_db = PineConeDB(embedding_dimension)
summary_gen = SummaryGen()
prompt_gen = PromptGen(embedding_model)
llm = LLMModel()

def send_initial_message(channel):
    initial_message = ("I am an AI chatbot coach for soft skills. Ask me for some advice."
                        "To finish the conversation, type 'exit'.")
    app.client.chat_postMessage(channel=channel, text=initial_message)

# Send an initial message when the app starts
send_initial_message(SLACK_CHANNEL_ID)


# Function to handle the message
@app.event("message")
def handle_message_events(event, say):
    user = event.get("user")  # User ID who sent the message
    user_input = event.get("text")  # User's message text
    channel = event.get("channel")

    # Only handle if there is a valid user input
    if user and user_input:
        if user_input.lower() == "exit":
            say("Take care, bye!", channel=channel)
            return  # Stop processing further

        session_conversation = "User: " + user_input

        # Get relevant context and generate prompt
        context = vector_db.give_relevant_summary(embedding_model.give_embedding(user_input))
        if len(context) > 0:
            context = context[0]
        else:
            context = ""
        prompt = prompt_gen.gen_prompt(user_input, context)

        # Get answer from LLM
        answer = llm.query_model(prompt)
        session_conversation += "\n Counselor: " + answer
        answer = answer + ". I hope that answers your question, you can ask me something else ..."
        # Send the response back to Slack
        say(answer, channel=channel)

        # Generate conversation summary and save
        conversation_summary = summary_gen.create_summary(session_conversation)
        conversation_embedding = embedding_model.give_embedding(conversation_summary)

        # Save the summary and embedding
        vector_db.save_summary(conversation_summary, conversation_embedding)

if __name__ == "__main__":
    # Start the app with the Socket Mode handler
    handler = SocketModeHandler(app, SLACK_APP_LEVEL_TOKEN)
    handler.start()
