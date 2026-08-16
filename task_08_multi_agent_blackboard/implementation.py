"""PDF Task 8: specialist-agent shared Redis-style blackboard with transactional persistence hooks."""
import threading

class Blackboard:
    def __init__(self): self.state={}; self.locks={}; self.log=[]
    def lock(self,key,agent):
        lock=self.locks.setdefault(key,threading.Lock()); ok=lock.acquire(timeout=5); return ok
    def unlock(self,key): self.locks[key].release()
    def commit(self,key,value,agent): self.state[key]=value; self.log.append({'agent':agent,'key':key,'value':value})

class SpecialistAgent:
    def __init__(self,name,role,bb): self.name=name; self.role=role; self.bb=bb
    def run(self,key,value):
        if not self.bb.lock(key,self.name): raise TimeoutError('deadlock/lock timeout')
        try: self.bb.commit(key,f'[{self.role}] {value}',self.name)
        finally: self.bb.unlock(key)

if __name__=='__main__':
    bb=Blackboard(); agents=[SpecialistAgent('code','CodeGenerator',bb),SpecialistAgent('audit','SystemAuditor',bb),SpecialistAgent('qa','QAAnalyst',bb)]
    for a,v in zip(agents,['generated code','security review passed','tests passed']): a.run('pipeline',v)
    print(bb.state); print(bb.log)
