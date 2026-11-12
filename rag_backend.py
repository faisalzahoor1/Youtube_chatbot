import os
import time
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# LOAD MODEL
load_dotenv()

# TRANSCRIPT
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["en"])
    transcript_list = transcript.to_raw_data()
    # print(transcript_list)
    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


# SPLITTER
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print(chunks[0].page_content)

# ADD EMBEDDINGS TO VECTOR STORE
# Embedding model: Google Gemini "gemini-embedding-001" (3072-dim vectors).
# Free tier allows ~100 embed requests/minute (each chunk counts as one),
# so we embed in batches and wait between them to stay under the limit.
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 90
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

texts = [c.page_content for c in chunks]
vectors = []
for i in range(0, len(texts), BATCH_SIZE):
    if i > 0:
        print("Waiting 60s to respect the free-tier rate limit...")
        time.sleep(60)
    vectors.extend(embeddings.embed_documents(texts[i:i + BATCH_SIZE]))
    # print(f"Embedded {len(vectors)}/{len(texts)} chunks with {EMBEDDING_MODEL}")

vector_store = FAISS.from_embeddings(list(zip(texts, vectors)), embeddings)

# RETRIEVER
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# result = retriever.invoke('What is deepmind')
# for res in result:
#     print(res.page_content)

# AUGMENTATION

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

# print(retrieved_docs)

# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# final_prompt = prompt.invoke({'context': context_text, 'question': question})

# answer = llm.invoke(final_prompt)
# print(answer)

# question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
# retrieved_docs    = retriever.invoke(question)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# parallel_chain.invoke('who is Demis')

parser = StrOutputParser()
main_chain = parallel_chain|prompt|llm|parser

main_chain.invoke('Can you summarize the video')