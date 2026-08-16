"""PDF Task 2: educational BPE + explicit causal LM training loop."""
from collections import Counter
import torch
import torch.nn as nn

class BPE:
    def __init__(self): self.merges=[]; self.vocab={}
    def train(self, texts, num_merges=50):
        words=[list(w)+['</w>'] for t in texts for w in t.split()]
        symbols=sorted(set(s for w in words for s in w)); self.vocab={s:i for i,s in enumerate(symbols)}
        for _ in range(num_merges):
            pairs=Counter((w[i],w[i+1]) for w in words for i in range(len(w)-1))
            if not pairs: break
            pair,_=pairs.most_common(1)[0]; merged=''.join(pair); self.merges.append(pair)
            for wi,w in enumerate(words):
                out=[]; i=0
                while i<len(w):
                    if i+1<len(w) and (w[i],w[i+1])==pair: out.append(merged); i+=2
                    else: out.append(w[i]); i+=1
                words[wi]=out
            self.vocab.setdefault(merged,len(self.vocab))
        return self
    def encode(self,text):
        ids=[]
        for word in text.split():
            pieces=list(word)+['</w>']
            for pair in self.merges:
                out=[]; i=0
                while i<len(pieces):
                    if i+1<len(pieces) and (pieces[i],pieces[i+1])==pair: out.append(''.join(pair)); i+=2
                    else: out.append(pieces[i]); i+=1
                pieces=out
            ids.extend(self.vocab[p] for p in pieces)
        return ids

class CausalLM(nn.Module):
    def __init__(self,vocab_size,d=128):
        super().__init__(); self.d=d; self.emb=nn.Embedding(vocab_size,d); self.proj=nn.Linear(d,vocab_size)
    def forward(self,x):
        n=x.size(1); mask=torch.triu(torch.full((n,n),float('-inf'),device=x.device),1)
        h=self.emb(x); a=torch.softmax(h@h.transpose(-1,-2)/self.d**0.5+mask,dim=-1); return self.proj(a@h)

if __name__=='__main__':
    texts=['transformer self attention','causal language model','generative artificial intelligence']
    b=BPE().train(texts,20); encoded=[b.encode(t) for t in texts]; maxlen=max(map(len,encoded)); data=torch.tensor([x+[0]*(maxlen-len(x)) for x in encoded],dtype=torch.long)
    model=CausalLM(len(b.vocab)); opt=torch.optim.AdamW(model.parameters(),lr=3e-3); loss_fn=nn.CrossEntropyLoss()
    for _ in range(3):
        opt.zero_grad(); logits=model(data[:,:-1]); loss=loss_fn(logits.reshape(-1,logits.size(-1)),data[:,1:].reshape(-1)); loss.backward(); opt.step()
    print('BPE vocabulary:',len(b.vocab),'training loss:',float(loss))
