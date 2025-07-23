from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class RadarConvFusion(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        assert obs_dim == 83, "Expected 72 lidar + 7 state inputs + 2 last action + 2 direction_info"

        # 雷达数据用 circular padding 卷积处理
        self.radar_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Flatten(),  # 输出 shape = [batch, 8×72]
        )

        self.linear = nn.Sequential(
            nn.Linear(16 * 72 + 9, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        radar = x[:, :72].unsqueeze(1)           # shape = [batch, 1, 72]
        state = x[:, 72:]                        # shape = [batch, 5]
        radar_feat = self.radar_conv(radar)      # shape = [batch, 16×72]
        concat = torch.cat([radar_feat, state], dim=1)
        return self.linear(concat)
