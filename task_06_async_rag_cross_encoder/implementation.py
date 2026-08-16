"""PDF Task 6: asynchronous dual-stage RAG service core.
Install: fastapi uvicorn faiss-cpu sentence-transformers torch transformers."""
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

class AsyncRAG:
    def __init__(self, documents):
        self.documents=documents
        self.bi=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.cross=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.emb=self.bi.encode(documents,normalize_embeddings=True)
    async def retrieve(self,query,k=5):
        q=self.bi.encode([query],normalize_embeddings=True)[0]
        scores=self.emb@q; idx=np.argsort(-scores)[:k]
        return [(self.documents[i],float(scores[i])) for i in idx]
    async def rerank(self,query,candidates,k=3):
        scores=self.cross.predict([(query,d) for d,_ in candidates])
        order=np.argsort(-np.asarray(scores))[:k]
        return [(candidates[i][0],float(scores[i])) for i in order]
    async def retrieve_and_rerank(self,query):
        return await self.rerank(query,await self.retrieve(query))

async def demo():
    rag=AsyncRAG(['Attention uses queries keys and values.','RAG retrieves external context.','LoRA trains low rank adapters.'])
    print(await rag.retrieve_and_rerank('How does retrieval augmented generation work?'))

if __name__=='__main__': asyncio.run(demo())
