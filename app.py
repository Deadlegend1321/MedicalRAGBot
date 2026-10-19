import os
import urllib.request
import traceback
import gradio as gr
from dotenv import load_dotenv

# 1. Load Local Environment Variables (API Keys) from .env file
load_dotenv()

# 2. Imports from LangChain (Modern, v0.3.x compatible standards)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# 3. Securely download the dataset locally if it does not exist
dataset_filename = 'ai-medical-chatbot.txt'
if not os.path.exists(dataset_filename):
    print("Downloading dataset...")
    url = "https://drive.google.com/uc?export=download&id=1727UCcg_Nxn9Nbj_LiX8pvGEpo591oxc"
    urllib.request.urlretrieve(url, dataset_filename)
    print("Dataset downloaded successfully!")

# 4. Load and Preprocess the Dataset
with open(dataset_filename, 'r', encoding='utf-8') as f:
    raw_data = f.read()

# Clean and separate sections based on delimiter
cleaned_data = raw_data.replace('\n\n', '\n').split('---')

# Standardize headings for clean retrieval parsing
for i in range(len(cleaned_data)):
    if i == 4:
        cleaned_data[i] = cleaned_data[i].replace('\n**', '\n###').replace('**', '')
    else:
        cleaned_data[i] = cleaned_data[i].replace('**', '')

# Organize into key-value context
ques_ans = {}
for section in cleaned_data:
    topics = section.split('\n###')
    for topic in topics[1:]:
        lines = topic.split('\n')
        if lines:
            ques_ans[lines[0].strip()] = " ".join(lines[1:]).strip()

# Aggregate all knowledge into a single text context separated by newlines
all_content = ""
for key, value in ques_ans.items():
    all_content += f"{key} {value}\n"

# 5. Chunk the text content
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=300,
    chunk_overlap=128,
    length_function=len
)
chunks = text_splitter.split_text(all_content)

# 6. Initialize Vector Store and Embeddings
embeddings = OpenAIEmbeddings()
INDEX_DIR = "faiss_index"

if os.path.exists(INDEX_DIR):
    vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)
retriever = vector_store.as_retriever(search_type="similarity", k=4)

# 7. Initialize Chat LLM Model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 8. Define Modern RAG Prompt Template
system_prompt = (
    "You are a medical assistant chatbot helping answer patient questions based only on the provided context.\n"
    "Do not guess or provide inaccurate information. If the answer is not found in the context, say you don’t know.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 9. Set Up Modern Retrieval Chain (Replaces legacy RetrievalQA)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 10. Gradio Predict Function
def predict(message, history):
    try:
        print(f"\n📥 Received query: {message}")

        # Execute modern retrieval chain
        response = rag_chain.invoke({"input": message})
        answer = response["answer"]

        print(f"✅ Generated Answer: {answer}")
        return answer
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ Exception occurred:\n{error_trace}")
        return "⚠️ Sorry, an internal error occurred. Make sure your OpenAI API Key is valid and active."

# 11. Launch Gradio Chatbot Interface
if __name__ == "__main__":
    gr.ChatInterface(
        fn=predict,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(
            placeholder="Ask me a question related to Healthcare and Medical Services",
            container=False,
            scale=7,
        ),
        title="Medical Document QA Chatbot (PyCharm)",
        examples=[
            "What are the main differences between acute and chronic medical conditions?",
            "Who needs Medical Consultation?",
            "Can patients seek medical consultation without insurance?",
        ],
    ).launch(share=False, theme=gr.themes.Soft())