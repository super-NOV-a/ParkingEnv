from parking_env2 import ParkingEnvArc          # 已写好的
from parking_env import ParkingEnv              # 原始（连续动作）
# from parking_env_disc import ParkingEnvDiscrete # 如果已有离散版

ENV_REGISTRY = {
    "arc":       ParkingEnvArc,
    "orig":      ParkingEnv,          # 连续 Box 动作
    # "orig_disc": ParkingEnvDiscrete,  # MultiDiscrete 动作
}
