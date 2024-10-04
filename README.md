# Soft Skills Conversational Bot

## Overview

This bot is designed to provide personalized soft skills utilizing a vector data base to add possible context of previous conversations in the session. The advice is specialized based on the user's question, the bot can offer targeted insights from various coaching perspectives, including:

- Feedback Techniques Coach
- Active Listening Coach
- Emotional Intelligence Coach
- Communication Skills Coach

## How It Works

1. **Initial Interaction**: The bot initiates the conversation by asking how it can help the user.

2. **User Inquiry**: When the user poses a question, the bot analyzes the query using vector embeddings to find the closest related conversation from its history.

3. **Contextual Prompt Creation**: Based on the user’s question together identified context, the bot creates a prompt, to be made later to an LLM, this prompt is made to match the query with the bot coaching style. This involves framing a narrative that includes:
   - Previous insights relevant to the user’s question
   - The user's current inquiry framed within that context


4. **Response Generation**: The prompt is then fed to a language model (LLM) to generate a response, the way that default model works is to make a completion of a story. So that when a new query comes, a short script is created using the relevant conversation history and the user's question.

For example, assuming the following situation:
   - User: "I feel that I don't have good communication within my team at work."
   - Generated Prompt: "In previous conversations, it was mentioned that effective communication is key for teamwork. A Communication Skills Coach, an expert in solving problems within teams for more effective communication, responds to the user: '..."
So that in this way the LLM can complete the story and a relevant reply can be extracted from this complete paragraph.   

5. **Summary and Storage**: After the LLM generates a response, the bot creates a summary of the interaction. This summary, along with the user’s question and the response, is stored in a vector database for future reference, when a context is searched.

## Features

- **Personalized Advice**: Tailors responses based on user input and past interactions.
- **Multi-faceted Coaching**: Access insights from different coaching areas.
- **Continuous Learning**: The bot retains information from interactions to enhance future advice.
- **User-friendly Interface**: Engaging and easy-to-use conversation flow.

## Getting Started

To interact with the Soft Skills Conversational Bot:

1. The configuration for Slack integration must be made. And the relevant keys must be added in the base code.
2. Start interacting with the Bot.
3. Receive tailored advice and insights.




## Connecting to Slack

Follow this instructions https://medium.com/@hamza-shafique/how-to-set-up-a-slack-bot-a-step-by-step-guide-19c8cc09ae34

More specifically: 

We have to creating a Slack App, configuring Permissions, enable Socket Mode, set Up Event Subscriptions
, get the Necessary Secrets and configure the Bot.

### Step1:Creating Slack App
On the Slack API website. Click on “Create New App” and choose “From Scratch". Select a name and the workspace for the Bot to be added. Hit that “Create App” button.

### Step2:Configure OAuth & Permissions
In your app’s dashboard, in “OAuth & Permissions” in the left-hand menu.
Scroll down to “Scopes” and add these ***Bot Token Scopes***:

- `channels:history`
- `channels:read`
- `chat:write`
- `groups:history`
- `im:history`
- `mpim:history`

Scroll back up and click “Install App to Workspace.” Follow the prompts to authorize your app. Copy the “Bot User OAuth Token” -to be used later.

### Step3:Enable Socket Mode
Socket Mode lets your bot connect to Slack’s APIs using WebSockets:

Find “Socket Mode” in the left-hand menu. Toggle the switch to enable it. Give your Socket Mode connection a name.
Make sure it has the `connections:write` permission.
Copy the “App Level Token” — another one to be used later.

### Step4:Set Up Event Subscriptions
This step tells Slack what events your bot should listen for:

Click on “Event Subscriptions” in the left-hand menu.
Toggle the switch to enable event subscriptions.
Under “Subscribe to Bot Events,” add these events:
-`message.channels` (for public channels)
-`message.groups` (for private channels)
-`message.im` (for direct messages)
-`message.mpim` (for multiparty direct messages)

### Step5:Obtain the Signing Secret
One last secret to grab:

Go to “Basic Information” in the left-hand menu. Scroll down to “App Credentials.” Copy the “Signing Secret” — to be used later.

### Configure Your Bot
Now, in the base code these variables must be set:

SLACK_BOT_TOKEN: The Bot User OAuth Token from Step 2
SLACK_SIGNING_SECRET: The Signing Secret from Step 5
SLACK_APP_LEVEL_TOKEN: The App Level Token from Step 3
SLACK_CHANNEL_ID: The ID of the Slack channel where your bot will operate (if applicable)
We will use the slack-bolt python package. It’s a framework to build Slack apps in a flash with the latest platform features.


## Text to voice

Text to voice to generate an answer back to the user is in `text2voice/text2voice.py` 
in this script an audio file is generated from a sample input. Here is documentation
of the library TTS https://docs.coqui.ai/en/latest/inference.html. This one requires `torch` and `TTS` that needs a conflicting version of `numpy` (greater than 2) so it has to be run separately.