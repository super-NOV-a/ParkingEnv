"""unified_policy.py – supports 4 architectures in a single registry
Network options
----------------
* **mlp**        : CustomMLP  (no RNN)
* **mlp_rnn**    : CustomMLP  + LSTM (RecurrentActorCriticPolicy)
* **conv**       : RadarConvFusion (no RNN)
* **conv_rnn**   : RadarConvExtractor + LSTM

Usage
-----
>>> model = make_model("conv_rnn", env, n_steps=128, batch_size=512)
>>> model.learn(5_000_000)
"""

from __future__ import annotations

import torch.nn as nn
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

# ---------------------------------------------------------------------------
# ① CustomMLP feature extractor (shared)
# ---------------------------------------------------------------------------
class CustomMLP(BaseFeaturesExtractor):    
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
        )

    def forward(self, obs):
        return self.net(obs)

# ---------------------------------------------------------------------------
# ③ RadarConvFusion (ring‑conv, no RNN)
# ---------------------------------------------------------------------------
class RadarConvFusion(BaseFeaturesExtractor):
    """1‑D circular conv on 72‑beam LiDAR + misc 11 dims → features_dim"""

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # assert observation_space.shape[0] == 83, "expect 72+11 obs"
        self.radar = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2, padding_mode="circular"),
            nn.ReLU(True),
            nn.Conv1d(8, 16, 3, padding=1, padding_mode="circular"),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),                      # (B,16)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 + 11, 256),
            nn.ReLU(True),
            nn.Linear(256, features_dim),
            nn.ReLU(True),
        )

    def forward(self, obs):
        lidar = obs[:, :-11].unsqueeze(1)            # (B,1,72)
        misc  = obs[:, -11:]                         # (B,11)
        return self.fc(torch.cat([self.radar(lidar), misc], dim=1))

# ---------------------------------------------------------------------------
# ④ RadarConvExtractor (conv for RNN)
# ---------------------------------------------------------------------------
class RadarConvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.lidar = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2, padding_mode="circular"),
            nn.ReLU(True),
            nn.Conv1d(16, 32, 3, padding=1, padding_mode="circular"),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 + 11, features_dim),
            nn.ReLU(True),
        )

    def forward(self, obs):
        return self.fc(torch.cat([self.lidar(obs[:, :-11].unsqueeze(1)), obs[:, -11:]], dim=1))

# ---------------------------------------------------------------------------
# ② CustomMlpLstmPolicy  (MLP + LSTM)
# ---------------------------------------------------------------------------
class CustomMlpLstmPolicy(RecurrentActorCriticPolicy):
    """全连接特征 + LSTM (用于 mlp_rnn)"""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # features_extractor_class=CustomMLP,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128], vf=[128]),
            lstm_hidden_size=128,
            n_lstm_layers=1,
            **kwargs,
        )

# ---------------------------------------------------------------------------
# ④ RadarLstmPolicy  (Conv + LSTM)
# ---------------------------------------------------------------------------
class RadarLstmPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_kwargs=dict(features_dim=64),
            net_arch=dict(pi=[128], vf=[128]),
            lstm_hidden_size=128,
            n_lstm_layers=1,
            **kwargs,
        )

# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------
POLICY_REGISTRY = {
    "mlp":       (PPO,          "MlpPolicy",        CustomMLP),
    "mlp_rnn":   (RecurrentPPO, CustomMlpLstmPolicy, CustomMLP),
    "conv":      (PPO,          "MlpPolicy",        RadarConvFusion),
    "conv_rnn":  (RecurrentPPO, RadarLstmPolicy,     RadarConvExtractor),
}


def make_model(model_type: str, env, **hyper):
    """Create PPO/RecurrentPPO model according to `model_type`.
    Parameters
    ----------
    model_type : str  in {mlp, mlp_rnn, conv, conv_rnn}
    env        : sb3 VecEnv
    hyper      : extra hyper‑parameters passed to algorithm constructor
    """
    algo_cls, policy_cls, feat_cls = POLICY_REGISTRY[model_type]
    policy_kwargs = dict(features_extractor_class=feat_cls)
    return algo_cls(policy_cls, env, policy_kwargs=policy_kwargs, **hyper)