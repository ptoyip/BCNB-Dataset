from torch import nn,mm,softmax
from torchvision import models

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Only use the CNN part and flatten part for vgg16_bn, without using the last classifier
        vgg16_bn = models.__dict__['vgg16_bn'](weights='DEFAULT')
        self.extractor = nn.Sequential(*(list(vgg16_bn.children())[:-1]))
        self.output_features_size = 512 * 7 * 7 # C * H * W
    
    def forward(self,x):
        feat = self.extractor(x)
        # print('feat.shape = ',feat.shape) # N * 512 * 7 * 7
        return feat
    
class FeatureAggregator(nn.Module):
    def __init__(self,cnn_output_features_size) -> None:
        super().__init__()
        self.pink_size = 256
        self.white_size = 128
        # FC: 1*(512*7*7) -> 256
        # Attention: 256 -> 1
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_features_size,self.pink_size),
            nn.Dropout(),
            nn.LeakyReLU()
        )
        # I dont understand why the attention layer should write like this.
        # Update: I've asked Prof Chen qifeng, he said attention just a term meaning when we doing feature selection, we
        # will learn the wait of different features, not necessary refer to the structure in "Attention is all you need".
        self.attention = nn.Sequential(
            nn.Linear(self.pink_size, self.white_size),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(self.white_size, 1),
            # Will the result be same if we ignore this dropout layer?
            nn.Dropout()
            # No need another activation function as we will have softmax later in the forward part.
        )
    def forward(self,x):
        x = x.view(-1, 512*7*7) # Flatten into N Column(512*7*7)
        x = self.fc(x) # N * (1 * 256)
        
        a = self.attention(x) #　N * 1
        a = a.T #should be N*1 -> 1*N i think???
        a = softmax(a,dim=1)
        
        # merge the origional feat to the feat after attention
        m = mm(a , x) # 1*N x N*128 = 1*128
        return m
    
class MIL(nn.Module):
    def __init__(self,):
        super(MIL, self).__init__()
        # vgg16_bn without last classification layer
        self.cnn = CNN()
        self.attention = FeatureAggregator(self.cnn.output_features_size)

        self.fc = nn.Sequential(
            nn.Linear(self.attention.pink_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64,2),
            #TODO: Sigmoid?
        )

    def forward(self, bag):
        feat = self.cnn(bag) # N * 3 * 256 * 256 -> N * 512 * 7 * 7
        post_attention_feat = self.attention(feat)
        x = self.fc(post_attention_feat)
        # print(x)
        return x