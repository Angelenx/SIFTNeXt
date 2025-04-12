import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TVF
from torchvision import transforms

class SIFT_Gaussion(nn.Module): # 用于生成高斯序列处理的图像
    """
    SIFT高斯金字塔生成模块 
    生成单个octave的5层高斯模糊图像（含原始图像）
    k: 尺度空间比例因子，默认值对应k=2^(1/3)
    sigma: 初始高斯核标准差
    """

    def __init__(self,k=1.148698354997035,sigma=1.6):
        super(SIFT_Gaussion,self).__init__()
        self.k = torch.Tensor([k])
        self.sigma = torch.Tensor([sigma])

    def cal_k_size(self,sigma):
        ks = int(6*sigma+1)
        if ks % 2 == 0:
            ks -=1 
        return ks
    
    def forward(self,x:torch.Tensor):
        
        s1=self.sigma.item()
        s2=(self.k*self.sigma).item()
        s3=(self.k.pow(2)*self.sigma).item()
        s4=(self.k.pow(3)*self.sigma).item()
        
        g0_1 = TVF.gaussian_blur(x, self.cal_k_size(s1), s1)
        g0_2 = TVF.gaussian_blur(g0_1, self.cal_k_size(s2), s2)
        g0_3 = TVF.gaussian_blur(g0_2, self.cal_k_size(s3), s3)
        g0_4 = TVF.gaussian_blur(g0_3, self.cal_k_size(s4), s4)
        
        g0_0 = x.unsqueeze(1)
        g0_1.unsqueeze_(1)
        g0_2.unsqueeze_(1)
        g0_3.unsqueeze_(1)
        g0_4.unsqueeze_(1)
        
        
        # (batch size,Gaussian_levels,channel,H,W)
        return torch.concatenate((g0_0,g0_1,g0_2,g0_3,g0_4),1)


class SIFT_DOG(nn.Module):
    def __init__(self,k=1.148698354997035,sigma=1.6):
        super(SIFT_DOG,self).__init__()
        self.gaussion_piramid = SIFT_Gaussion(k,sigma)
        self.gray = transforms.Grayscale(1)
        
    def sift_downsample(self,x):
        return x[:, :, ::2, ::2]
    
    def forward(self,p0):
        # (batch-size,3,512,512) input
        p0 = self.gray(p0) # (batchSize,channel,H,W)
        p0_g = self.gaussion_piramid(p0)
        
        p1 = self.sift_downsample(p0_g[:,2])
        p1_g = self.gaussion_piramid(p1)
        
        p2 = self.sift_downsample(p1_g[:,2])
        p2_g = self.gaussion_piramid(p2)
        
        p3 = self.sift_downsample(p2_g[:,2])
        p3_g = self.gaussion_piramid(p3)
        
        p4 = self.sift_downsample(p3_g[:,2])
        p4_g = self.gaussion_piramid(p4)
        
        p0_d = torch.diff(p0_g, n=1, dim=1)
        p1_d = torch.diff(p1_g, n=1, dim=1)
        p2_d = torch.diff(p2_g, n=1, dim=1)
        p3_d = torch.diff(p3_g, n=1, dim=1)
        p4_d = torch.diff(p4_g, n=1, dim=1)

        
        return p0_d,p1_d,p2_d,p3_d,p4_d # 返回高斯差分金字塔

class CBAM(nn.Module):
    def __init__(self):
        super(CBAM,self).__init__()
        
    def forward(self,x):
        pass
    
class SRE_CONV(nn.Module):
    def __init__(self):
        super(SRE_CONV,self).__init__()
        
    def forward(self,x):
        pass

class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt,self).__init__()
        
    def forward(self,x):
        pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    testnn = SIFT_DOG()
    x = torch.Tensor(plt.imread('SIFT_CNN\ISIC_0000002.jpg')).permute(2,0,1)/255
    x.unsqueeze_(0)
    x = TVF.resize(x,(512,512))
    # plt.imshow(x[0].permute(1,2,0).numpy())
    # plt.show()
    # 模拟1张512x512的RGB图像
    y = testnn(x)
    i = 0
    for it in y:
        for j in range(it.shape[1]):
            img = ((it[0,j,0])).numpy()
            plt.imsave('./差分金字塔/%d_%d.jpg'%(i,j),img)
        i=i+1
    pass

