from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.prompts import ChatPromptTemplate

import os
# os.makedirs("flagged", exist_ok=True)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
Imagine yourself as a cricket expert and answer the questions based on context provided.
Give a brief summary for user question, dont copy answer from the context and use it as reference.
Answer should not be too short of too long.
Answer only if you are so sure, dont hallucinate and answer about the teams user asked.
Also, dont include metadata like matchID, seriesID while answering, just answer like an cricket expert.
You are answering about IPL 2024, answer based on that. 
When user refers a number, thats mostly in the perspective of IPL, not related to the context. Use your IPL knowledge in those cases.
If there are multiple matches, answer both matches seperately in two paragraphs and most importantly dont answer about 2nd match if you are not 100 percent confident.
Final answer should include who won the match, key contributions. If user query is about a match.
: {question}
"""

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = VertexAIEmbeddings(
        model_name="textembedding-gecko",
        batch_size=1,
        requests_per_minute=60
    )
    db = Chroma(persist_directory='chroma_db', embedding_function=embedding_function)

    # print(len(db.get()['ids']))
    

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)
    documents = []
    for doc,_score in  results:
        documents.append(doc.metadata["source"].split('\\')[-1])
    print(documents)

    docs = []
    for document in documents:
        docs.append(document+"1")
        docs.append(document+"2")

    doc_result = db.get(ids=docs)
    query_docs = doc_result["documents"]

    context_text = "\n\n---\n\n".join([doc for doc in query_docs])
    return context_text

def llm_answer(context_text,query_text):
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)

  import vertexai
  from vertexai.preview.generative_models import GenerativeModel, ChatSession

  project_id = "myprojectrag"
  location = "us-central1"
  vertexai.init(project=project_id, location=location)

  model = GenerativeModel("gemini-1.5-pro")
  chat = model.start_chat()

  def get_chat_response(chat: ChatSession, prompt: str):
      response = chat.send_message(prompt)
      return response.text

  return get_chat_response(chat, prompt)


# query_text = "RCB vs SRH detailed score card"
def llm_chat(query_text):
  context_text = query_rag(query_text)
  return llm_answer(context_text,query_text)

import gradio as gr


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Gradio Application</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .welcome-message {
            background-color: #e9f7fe;
            border: 1px solid #b3e5fc;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 20px;
        }
        #gradio-app {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>My Gradio Application</h1>
        <div class="welcome-message">
            <p>Welcome to IPL 2024 RAG application! Please enter your query regarding IPL 2024.</p>
        </div>
        <div id="gradio-app"></div>
    </div>
</body>
</html>
"""

demo = gr.Interface(
    fn=llm_chat,
    inputs=["text"],
    outputs=["text"],

    title="IPL 2024 RAG application",
    description="I can answer questions about IPL 2024.\n Please enter any two teams to find the result. (Example, RCB vs KKR)",
    allow_flagging="never" 
)

# demo.launch()

from fastapi import FastAPI


app = FastAPI()

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    # uvicorn.run(app, host="0.0.0.0", port=port)
    uvicorn.run(app)