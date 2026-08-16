"""PDF Task 12: CLIP-style teacher/student composite distillation loss."""
import torch
import torch.nn.functional as F

def distillation_loss(student_logits,teacher_logits,T=2.0,alpha=0.5):
    kl=F.kl_div(F.log_softmax(student_logits/T,-1),F.softmax(teacher_logits/T,-1),reduction='batchmean')*(T*T)
    cosine=(1-F.cosine_similarity(student_logits,teacher_logits).mean())
    return alpha*cosine+(1-alpha)*kl,cosine,kl

if __name__=='__main__':
    teacher=torch.randn(8,512); student=torch.randn(8,512,requires_grad=True)
    loss,cos,kl=distillation_loss(student,teacher); loss.backward(); print('loss/cosine/KL:',float(loss),float(cos),float(kl))
