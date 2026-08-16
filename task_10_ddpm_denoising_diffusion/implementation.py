"""PDF Task 10: DDPM forward schedule and reverse denoising loop interface."""
import torch

def schedule(T=1000,beta_start=1e-4,beta_end=0.02):
    beta=torch.linspace(beta_start,beta_end,T); alpha=1-beta; abar=torch.cumprod(alpha,0); return beta,alpha,abar

def q_sample(x0,t,abar,noise=None):
    noise=torch.randn_like(x0) if noise is None else noise
    a=abar[t].view(-1,*([1]*(x0.ndim-1))); return a.sqrt()*x0+(1-a).sqrt()*noise

def reverse_ddpm(model,shape,device='cpu',T=1000):
    beta,alpha,abar=schedule(T); x=torch.randn(shape,device=device)
    for t in range(T-1,-1,-1):
        tb=torch.full((shape[0],),t,device=device,dtype=torch.long)
        pred_noise=model(x,tb)
        a=alpha[t].to(device); ab=abar[t].to(device); b=beta[t].to(device)
        mean=(x-(b/(1-ab).sqrt())*pred_noise)/a.sqrt()
        x=mean if t==0 else mean+b.sqrt()*torch.randn_like(x)
    return x

if __name__=='__main__':
    _,_,abar=schedule(); x=torch.randn(1,1,28,28)
    for t in [0,250,500,999]: print(t,float(abar[t]),tuple(q_sample(x,torch.tensor([t]),abar).shape))
    print('Reverse loop is ready for a trained epsilon-prediction model.')
