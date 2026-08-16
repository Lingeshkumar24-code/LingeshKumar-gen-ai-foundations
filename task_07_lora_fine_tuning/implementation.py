"""PDF Task 7: LoRA block for frozen Transformer projections."""
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self,in_features,out_features,r=8,alpha=16):
        super().__init__(); self.base=nn.Linear(in_features,out_features); self.base.requires_grad_(False)
        self.A=nn.Parameter(torch.randn(r,in_features)*0.01); self.B=nn.Parameter(torch.zeros(out_features,r)); self.scale=alpha/r
    def forward(self,x): return self.base(x)+(x@self.A.T@self.B.T)*self.scale

def train_step(layer,x,target,opt):
    opt.zero_grad(); loss=((layer(x)-target)**2).mean(); loss.backward(); opt.step(); return float(loss)

if __name__=='__main__':
    layer=LoRALinear(256,256,r=8); opt=torch.optim.AdamW([layer.A,layer.B],lr=1e-3)
    x=torch.randn(8,256); y=torch.randn(8,256)
    print('base trainable:',sum(p.numel() for p in layer.base.parameters() if p.requires_grad),'LoRA params:',sum(p.numel() for p in (layer.A,layer.B)))
    for _ in range(3): print('loss:',train_step(layer,x,y,opt))
