import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TVF
from torchvision import transforms

class SIFT_Gaussion(nn.Module): # 用于生成高斯序列处理的图像
    """
    SIFT高斯金字塔生成模块 
    功能：生成单个octave的5层高斯模糊图像（含原始图像）
    参数说明：
        k: 尺度空间比例因子，默认值对应k=2^(1/3)
        sigma: 初始高斯核标准差
    """
    
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
        # (batchsize,3,512,512) input
        p0 = self.gray(p0) # (batchSize,channel,H,W)
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

        
        return p0_d,p1_d,p2_d,p3_d,p4_d # 返回高斯差分金字塔
    
if __name__ == '__main__':
    testnn = SIFT_DOG()
    x = torch.zeros((16,3,512,512))# 模拟16张512x512的RGB图像
    y = testnn(x)
    pass

