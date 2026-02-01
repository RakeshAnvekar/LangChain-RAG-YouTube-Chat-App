# ===================== Imports =====================
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

# ===================== Load ENV =====================
load_dotenv()

# ===================== STEP 1: Fetch YouTube Transcript =====================
video_id = "F2FmTdLtb_4"  

try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)

    # Convert transcript to single string
    transcript = " ".join(snippet.text for snippet in transcript_list)

    print("Transcript fetched successfully.")
    print(transcript[:500])

except TranscriptsDisabled:
    raise Exception("Transcripts are disabled for this video.")

# ===================== STEP 2: Text Splitting =====================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])
print(f"Total Chunks Created: {len(chunks)}")

# ===================== STEP 3: Embeddings + Vector Store =====================
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embedding=embeddings)

print("Embeddings generated and stored in FAISS.")

# ===================== STEP 4: Retriever =====================
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("Retriever created.")

# ===================== STEP 5: LLM =====================
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)

# ===================== STEP 6: Prompt =====================
prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.
Answer the question only based on the transcript context below.
If the answer is not found in the context, say "I don't know".

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)
# ===================== STEP 7: Create Chain =====================
parallel_chain = RunnableParallel(
        {
           "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    


main_chain= parallel_chain | prompt | llm | StrOutputParser()
# ===================== STEP 8: Ask Question =====================
question = "What is reliability in system design?"

answer = main_chain.invoke(question)

print("\n####################   Answer   ####################\n")
print(answer)
