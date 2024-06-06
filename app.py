from flask import Flask, render_template, request, jsonify
import os
import openai
from dotenv import load_dotenv
import requests

app = Flask(__name__)
# Load environment variables from .env file
load_dotenv()
import os
from openai import AzureOpenAI


endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
# set the deployment name for the model we want to use
deployment = "gpt-35-turbo-16k"   

client = AzureOpenAI(
    base_url=f"{endpoint}/openai/deployments/{deployment}/extensions",
    api_version="2023-09-01-preview"
)


message_history = []# message history
# initial message
message_text = [{"role":"system","content":"You are an AI assistant that helps people by answering the questions asked."}]
message_history.extend(message_text)
# Initiate Flask routes for chatbot
@app.route('/')
def index():
    return render_template('index.html')

# Chatbot route--main route for chatbot--handels both GET and POST requests
@app.route('/chat', methods=['GET', 'POST'])
def chatbot():
        
        user_message = request.form.get('message')
        message_history.append({"role":"user","content":user_message})
        completion = client.chat.completions.create(
        messages=message_history,
        model=deployment,
        temperature=0,
        top_p=1,
        max_tokens=800,
        stop=None,
        stream=False,
        extra_body={
            "dataSources": [
                {
                    "type": "AzureCognitiveSearch",
                    "parameters": {
                        "endpoint": os.environ["SEARCH_ENDPOINT"],
                        "key": os.environ["SEARCH_KEY"],
                        "indexName": os.environ["SEARCH_INDEX_NAME"],
                        "top_n_documents": 1,
                        "role_information": "You are an AI assistant that helps people find information.",
                    }
                }
            ]
        }
    )
        message_history.append({"role":"assistant","content":completion.choices[0].message.content})

        return jsonify({"response": completion.choices[0].message.content})


if __name__ == '__main__':
    app.run(debug=True)