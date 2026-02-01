from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
from dotenv import load_dotenv
import os
load_dotenv()


################# step 1: Fetch YouTube Transcript  [Indexing --Document Ingestion]######################

video_Id="F2FmTdLtb_4"  # Replace with your YouTube video ID
try: 
    #we get the transcript based on time-stamped segments
    #[{`text`: `some text`, `start`: 0.0, `duration`: 4.0}, {...}, ...]
    ytt_api = YouTubeTranscriptApi()
    transcript_list=ytt_api.fetch(video_Id)

    #flatten the list of dictionaries to a single string

    for snippet in transcript_list:
        transcript = " ".join(snippet.text for snippet in transcript_list)
        print("Transcript fetched successfully.")
        print(transcript[:500])  # Print first 500 characters of the transcript

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")


################### step 2: Text Splitting ##########################
# Initialize text splitter
splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
# Create document chunks
chunks=splitter.create_documents([transcript])
print(f"Total Chunks Created: {len(chunks)}")

################### step 3: Generate Embeddings and Store in Vector DB ##########################
# Initialize OpenAI Embeddings
embeddings=OpenAIEmbeddings()
vector_sore=FAISS.from_documents(chunks,embedding=embeddings)
print("Embeddings generated and stored in FAISS vector store.")

################### step 4: Create a Retrieval  ##########################
retriever=vector_sore.as_retriever(search_type="similarity",search_kwargs={"k":3})# it will return top 3 similar chunks or documents
print("Retriever created.")



################### step 5: llm ##########################

llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)

###########step 6: prompt template ##########################

prompt=PromptTemplate(
template=""" 
You are a helpful AI assistant
Answer the question only based on the transcript context below.
if the answer is not found in the context, say "I don't know".
{context}
Question: {question}
""",
input_variables=["context","question"]
)

question="What is relability in system design?"
docs=retriever.invoke(question)
context="\n".join([doc.page_content for doc in docs])

print("Context for the question:")
print(context)

######## step 7: Generate Answer ##########################
final_prompt=prompt.invoke({"context": context, "question": question})
response=llm.invoke(final_prompt)
print("####################   Answer   ####################")
print(response.content)