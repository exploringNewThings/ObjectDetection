from .transforms import BetaTransforms
from torchvision.transforms import transforms

#transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
class Normalize(BetaTransforms):
    def __init__(self,mean,std):
        super(Normalize, self).__init__("Normalize")
        self.mean = mean
        self.std = std
    
    def transform(self,sample):
        img = sample['image']
        img = img.astype('float')
        img[:,:,0] = (img[:,:,0]-self.mean[0])/self.std[0]
        img[:,:,1] = (img[:,:,1]-self.mean[1])/self.std[1]
        img[:,:,2] = (img[:,:,2]-self.mean[2])/self.std[2]
        
        sample['image'] = img
        
        return sample
