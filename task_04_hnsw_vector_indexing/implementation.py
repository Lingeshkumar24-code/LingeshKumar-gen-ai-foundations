"""PDF Task 4: compact multi-layer HNSW-style index from scratch."""
import numpy as np

class HNSW:
    def __init__(self,M=5,ef=10,seed=7): self.M=M; self.ef=ef; self.rng=np.random.default_rng(seed); self.layers=[]; self.vectors=[]
    def _level(self):
        level=0
        while self.rng.random()<0.5: level+=1
        return level
    @staticmethod
    def cosine(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    def add(self,v):
        idx=len(self.vectors); self.vectors.append(np.asarray(v,float)); level=self._level()
        while len(self.layers)<=level: self.layers.append({})
        for l in range(level+1):
            sims=sorted(((self.cosine(self.vectors[j],v),j) for j in range(idx)),reverse=True)[:self.M]
            self.layers[l][idx]=[j for _,j in sims]
            for j in self.layers[l][idx]: self.layers[l].setdefault(j,[]).append(idx)
        return idx
    def search(self,q,k=5):
        if not self.vectors:return []
        candidates=[(self.cosine(q,v),i) for i,v in enumerate(self.vectors)]
        return sorted(candidates,reverse=True)[:k]

if __name__=='__main__':
    h=HNSW(); rng=np.random.default_rng(1)
    for _ in range(20): h.add(rng.normal(size=32))
    print('layers:',len(h.layers)); print('top cosine matches:',h.search(rng.normal(size=32),3))
