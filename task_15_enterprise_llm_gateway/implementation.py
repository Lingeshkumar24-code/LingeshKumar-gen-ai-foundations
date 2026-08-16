"""PDF Task 15: FastAPI-compatible token bucket and provider failover core."""
import time

class TokenBucket:
    def __init__(self,capacity=3,rate=1.0): self.capacity=capacity; self.rate=rate; self.tokens=float(capacity); self.last=time.monotonic()
    def allow(self):
        now=time.monotonic(); self.tokens=min(self.capacity,self.tokens+(now-self.last)*self.rate); self.last=now
        if self.tokens>=1: self.tokens-=1; return True
        return False

class ProviderRouter:
    def __init__(self): self.primary_healthy=True
    def route(self,prompt): return 'primary' if self.primary_healthy else 'secondary'

if __name__=='__main__':
    b=TokenBucket(3,1); r=ProviderRouter()
    for i in range(5): print(i,b.allow(),r.route('test'))
    r.primary_healthy=False; print('fallback:',r.route('test'))
