"""PDF Task 5: manual backward calculus for single-head attention."""
import numpy as np

def softmax(x):
    e=np.exp(x-x.max(axis=-1,keepdims=True)); return e/e.sum(axis=-1,keepdims=True)

def forward_backward(X,Wq,Wk,Wv,target):
    n,d=X.shape; dk=Wq.shape[1]
    Q=X@Wq; K=X@Wk; V=X@Wv; S=Q@K.T/np.sqrt(dk); A=softmax(S); O=A@V
    dO=2*(O-target)/(O.size); dV=A.T@dO; dA=dO@V.T
    dS=A*(dA-(dA*A).sum(axis=-1,keepdims=True)); dS/=np.sqrt(dk)
    dQ=dS@K; dK=dS.T@Q
    return O,dQ,dK,dV,X.T@dQ,X.T@dK,X.T@dV

if __name__=='__main__':
    rng=np.random.default_rng(7); X=rng.normal(size=(32,16)); Wq=Wk=Wv=rng.normal(size=(16,16)); target=rng.normal(size=X.shape)
    O,dQ,dK,dV,dWq,dWk,dWv=forward_backward(X,Wq,Wk,Wv,target)
    print('projection gradient norms:',[float(np.linalg.norm(g)) for g in (dWq,dWk,dWv)])
