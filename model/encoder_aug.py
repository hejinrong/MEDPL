import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# load hyperparameters

class StochasticFeatureAugmentation(nn.Module):
    def __init__(self, noise_type='independent', dim_emb=160, num_classes=10):
        super(StochasticFeatureAugmentation, self).__init__()
        self.noise_type = noise_type
        self.dim_emb = dim_emb
        self.num_classes = num_classes
        self.covariances = nn.ParameterList(
            [nn.Parameter(torch.eye(self.dim_emb) * 0.1) for _ in range(num_classes)])  # 假设特征维度为128

    def forward(self, features, labels):
        # 生成独立噪声
        if self.noise_type == 'independent':
            alpha = torch.normal(mean=1.0, std=0.1, size=features.size()).to(features.device)
            beta = torch.normal(mean=0.0, std=0.1, size=features.size()).to(features.device)
            augmented_features = alpha * features + beta

        elif self.noise_type == 'adaptive':
            augmented_features = features.clone()
            for i in range(self.num_classes):
                mask = (labels == i)  # 创建布尔张量
                if torch.any(mask):  # 检查是否有任何为 True 的元素
                    num_selected_samples = mask.sum().item()  # 获取选中样本的数量

                    # 提取标准差，生成形状为 (num_selected_samples, 128)
                    std = self.covariances[i].sqrt().diag()  # 取对角线作为标准差
                    noise = torch.normal(mean=0.0, std=std.expand(num_selected_samples, -1)).to(features.device)

                    # 将噪声加到选中的样本上
                    augmented_features[mask] += noise

        return augmented_features


# class StochasticFeatureAugmentation(nn.Module):
#     def __init__(self, noise_type='independent', dim_emb=160, num_classes=10):
#         super(StochasticFeatureAugmentation, self).__init__()
#         self.noise_type = noise_type
#         self.dim_emb = dim_emb
#         self.num_classes = num_classes
#         self.covariances = nn.ParameterList(
#             [nn.Parameter(torch.eye(self.dim_emb) * 0.1) for _ in range(num_classes)])  # 假设特征维度为128
#
#     def forward(self, features, labels):
#         # 生成独立噪声
#         if self.noise_type == 'independent':
#             alpha = torch.normal(mean=1.0, std=0.1, size=features.size()).to(features.device)
#             beta = torch.normal(mean=0.0, std=0.1, size=features.size()).to(features.device)
#             augmented_features = alpha * features + beta
#
#         elif self.noise_type == 'adaptive':
#             augmented_features = features.clone()
#             for i in range(self.num_classes):
#                 mask = (labels == i)  # 创建布尔张量
#                 if torch.any(mask):  # 检查是否有任何为 True 的元素
#                     num_selected_samples = mask.sum().item()  # 获取选中样本的数量
#
#                     # 创建多元正态分布对象
#                     mean = torch.zeros(self.dim_emb).to(features.device)
#                     multivariate_normal = MultivariateNormal(loc=mean, covariance_matrix=self.covariances[i].to(features.device))
#                     # 生成噪声样本
#                     noise = multivariate_normal.sample().to(features.device)
#
#                     augmented_features[mask] += noise
#
#         return augmented_features


class SpectralEncoder(nn.Module):
    def __init__(self, input_channels, patch_size, feature_dim):
        super(SpectralEncoder, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.inter_size = 24

        self.conv1 = nn.Conv3d(1, self.inter_size, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
                               bias=True)
        self.bn1 = nn.BatchNorm3d(self.inter_size)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), padding_mode='zeros', bias=True)
        self.bn2 = nn.BatchNorm3d(self.inter_size)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), padding_mode='zeros', bias=True)
        self.bn3 = nn.BatchNorm3d(self.inter_size)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv3d(self.inter_size, self.feature_dim,
                               kernel_size=(((self.input_channels - 7 + 2 * 1) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(self.feature_dim)
        self.activation4 = nn.ReLU()

       # self.attention = LocalAttention(128, 7, 7)

        #self.avgpool = nn.AvgPool3d((1, self.patch_size, self.patch_size))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))

        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
      #  x1 = self.attention(x1)

        x1 = self.avgpool(x1)
        x1 = x1.reshape((x1.size(0), -1))

        return x1


class SpatialEncoder(nn.Module):
    def __init__(self, input_channels, patch_size, feature_dim):
        super(SpatialEncoder, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.inter_size = 24

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, self.inter_size, kernel_size=(self.input_channels, 1, 1))
        self.bn5 = nn.BatchNorm3d(self.inter_size)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv8 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 1, 1))

        self.conv6 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn6 = nn.BatchNorm3d(self.inter_size)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn7 = nn.BatchNorm3d(self.inter_size)
        self.activation7 = nn.ReLU()

        #self.avgpool = nn.AvgPool3d((1, self.patch_size, self.patch_size))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(self.inter_size, out_features=self.feature_dim))


    def forward(self, x):
        x = x.unsqueeze(1)

        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
      #  x2 = self.attention(x2)


        x2 = self.avgpool(x2)
        x2 = x2.reshape((x2.size(0), -1))

        x2 = self.fc(x2)

        return x2


class Encoder(nn.Module):
    def __init__(self, n_dimension, patch_size, emb_size,class_num, dropout=0.5):
        super(Encoder, self).__init__()
        self.n_dimension = n_dimension
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.class_num = class_num
        self.dropout = dropout

        self.spectral_encoder = SpectralEncoder(input_channels=self.n_dimension, patch_size=self.patch_size, feature_dim=self.emb_size)
        self.spatial_encoder = SpatialEncoder(input_channels=self.n_dimension, patch_size=self.patch_size, feature_dim=self.emb_size)
        self.sfa = StochasticFeatureAugmentation(noise_type='adaptive',dim_emb = self.emb_size, num_classes=self.class_num)

    def forward(self, x, s_or_q="query"):
        labels = torch.tensor(list(range(self.class_num)))
        spatial_feature = self.spatial_encoder(x)
        spectral_feature = self.spectral_encoder(x)
        spatial_spectral_fusion_feature = 0.5 * spatial_feature + 0.5 * spectral_feature
        if self.training and s_or_q == "support":
            augmentation_feature = self.sfa(spatial_spectral_fusion_feature, labels)
            return spatial_spectral_fusion_feature,augmentation_feature

        return spatial_spectral_fusion_feature