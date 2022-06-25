import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_shape, std=0.05, image_size=1024):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = torch.zeros(self.shape).cuda()
        self.std = std
        self.image_size=image_size
    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        try:
            return x + self.noise
        except:
            self.noise = torch.zeros((x.size(0),) + (1, self.image_size, self.image_size)).cuda()
            self.noise.data.normal_(0, std=self.std)
            return x + self.noise

class Encoder(nn.Module):
    def __init__(self, batch_size, std, noise, input_shape = (1, 128, 128), p=0.5, data='xray', cudable=True):
        super(Encoder, self).__init__()
        self.std = std
        self.noise = noise
        self.gn = GaussianNoise(batch_size, input_shape=input_shape, std=self.std)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)
        self.data = data

        if data == 'xray':
            self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.neck = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 4 * 4, 4096),  # input_shape=[128, 128]
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 256),
        )
 
    def forward(self, x, label=None, is_matching=False, T=100):
        if self.noise and self.training:
            x = self.gn(x)
        x = self.bn1(self.act(self.conv1(x)))
        x = self.bn2(self.act(self.conv2(x)))
        x = self.bn3(self.act(self.conv3(x)))
        x = self.bn4(self.act(self.conv4(x)))
        x = self.bn5(self.act(self.conv5(x)))

        x = x.view(-1, 128 * 4 * 4)  # input_shape=[128, 128]
        feature = self.neck(x)
        
        return feature
    

class AlexNet(nn.Module):
    def __init__(self, batch_size, n_classes, std, noise, input_shape = (1, 128, 128), p=0.5, mom=0.999, data='xray', cudable=True):
        super(AlexNet, self).__init__()
        if data == 'xray':
            self.n_class = n_classes #14
        else:
            self.n_class = n_classes #7
        self.data = data
        
        self.mom = mom
        self.K = batch_size // self.n_class   # memory size
        self.cudable = cudable
        self.encoder_student = Encoder(batch_size, std, noise, input_shape, p, data, cudable)
        
        self.fc9 = nn.Sequential(
            nn.Linear(256, self.n_class),
        )

    @torch.no_grad()
    def _momentum_update_teacher_encoder(self):
        """
        Momentum update of the teacher encoder
        """
        for param_s, param_t in zip(self.encoder_student.parameters(), self.encoder_teacher.parameters()):
            param_t.data = param_t.data * self.mom + param_s.data * (1. - self.mom)
        
    def forward(self, x, out_feat=False):        
        feature = self.encoder_student(x)
        score = self.fc9(torch.nn.functional.relu(feature))
        if self.data == 'xray':
            pred = torch.sigmoid(score)
            return feature, score, pred
        else:
            if out_feat:
                return score, feature   
            else:
                return score
        