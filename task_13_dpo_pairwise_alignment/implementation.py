"""PDF Task 13: DPO loss for policy/reference and chosen/rejected pairs."""
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen,policy_rejected,ref_chosen,ref_rejected,beta=0.1):
    margin=(policy_chosen-policy_rejected)-(ref_chosen-ref_rejected)
    return (-F.logsigmoid(beta*margin)).mean(),margin.detach()

if __name__=='__main__':
    pc=torch.tensor([-1.2,-.8],requires_grad=True); pr=torch.tensor([-2.5,-3.1]); rc=torch.tensor([-1.5,-1.0]); rr=torch.tensor([-2.2,-2.8])
    loss,margin=dpo_loss(pc,pr,rc,rr); loss.backward(); print('margins:',margin.tolist(),'loss:',float(loss))
