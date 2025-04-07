import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TVF
from torchvision import transforms

class SIFT_Gaussion(nn.Module):
    def __init__(self,k=1.148698354997035,sigma=1.6):
        super(SIFT_Gaussion,self).__init__()
        self.k = torch.Tensor([k])
        self.sigma = torch.Tensor([sigma])
        
    def forward(self,x:torch.Tensor):
        g0_1 = TVF.gaussian_blur(x, 3, self.sigma.item())
        g0_2 = TVF.gaussian_blur(g0_1, 3, (self.k*self.sigma).item())
        g0_3 = TVF.gaussian_blur(g0_2, 3, (self.k.pow(2)*self.sigma).item())
        g0_4 = TVF.gaussian_blur(g0_3, 3, (self.k.pow(3)*self.sigma).item())
        
        g0_0 = x.unsqueeze(1)
        g0_1.unsqueeze_(1)
        g0_2.unsqueeze_(1)
        g0_3.unsqueeze_(1)
        g0_4.unsqueeze_(1)
        
        
        # (batchsize,Gaussion_levels,channel,H,W)
        return torch.concatenate((g0_0,g0_1,g0_2,g0_3,g0_4),1)


class SIFT_DOG(nn.Module):
    def __init__(self,k=1.148698354997035,sigma=1.6):
        super(SIFT_DOG,self).__init__()
        self.gaussion_piramid = SIFT_Gaussion(k,sigma)
        self.gray = transforms.Grayscale(1)
        
    
    def forward(self,p0):
        # (3,512,512) input
        p0 = self.gray(p0) #(batchSize,channel,H,W)
        p0_g = self.gaussion_piramid(p0)
        
        p1 = F.avg_pool2d(p0_g[2],2,2)
        p1_g = self.gaussion_piramid(p1)
        
        p2 = F.avg_pool2d(p1_g[2],2,2)
        p2_g = self.gaussion_piramid(p2)
        
        p3 = F.avg_pool2d(p2_g[2],2,2)
        p3_g = self.gaussion_piramid(p3)
        
        p4 = F.avg_pool2d(p3_g[2],2,2)
        p4_g = self.gaussion_piramid(p4)
        
        p0_d = torch.diff(p0_g, n=1, dim=1)
        p1_d = torch.diff(p1_g, n=1, dim=1)
        p2_d = torch.diff(p2_g, n=1, dim=1)
        p3_d = torch.diff(p3_g, n=1, dim=1)
        p4_d = torch.diff(p4_g, n=1, dim=1)

        
        return p0_d,p1_d,p2_d,p3_d,p4_d
    
if __name__ == '__main__':
    testnn = SIFT_DOG()
    x = torch.zeros((16,3,512,512))
    y = testnn(x)
    pass

