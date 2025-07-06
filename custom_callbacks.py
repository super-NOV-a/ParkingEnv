from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CustomLoggingCallback(BaseCallback):
    """
    自定义回调函数，用于记录额外指标
    """

    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        # 存储每个环境的指标
        self.episode_rewards = []
        self.episode_lengths = []
        self.curriculum_levels = []
        self.min_distances = []
        self.target_distances = []

        # 用于临时存储当前回合的数据
        self.current_rewards = {}
        self.current_lengths = {}
        self.current_levels = {}
        self.current_min_dists = {}
        self.current_target_dists = {}

    def _on_training_start(self) -> None:
        # 初始化字典
        for i in range(self.training_env.num_envs):
            self.current_rewards[i] = 0.0
            self.current_lengths[i] = 0
            self.current_levels[i] = 0
            self.current_min_dists[i] = float('inf')
            self.current_target_dists[i] = float('inf')

    def _on_step(self) -> bool:
        # 获取当前环境信息
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [])

        for env_idx in range(self.training_env.num_envs):
            # 累加奖励
            self.current_rewards[env_idx] += rewards[env_idx]
            self.current_lengths[env_idx] += 1

            # 更新其他指标
            if env_idx < len(infos) and infos[env_idx]:
                info = infos[env_idx]
                if 'curriculum_level' in info:
                    self.current_levels[env_idx] = info['curriculum_level']
                if 'min_distance' in info:
                    self.current_min_dists[env_idx] = min(
                        self.current_min_dists[env_idx], info['min_distance'])
                if 'target_distance' in info:
                    self.current_target_dists[env_idx] = info['target_distance']

                # 检查回合是否结束
                if 'episode' in info:
                    # 存储完成的回合数据
                    self.episode_rewards.append(self.current_rewards[env_idx])
                    self.episode_lengths.append(self.current_lengths[env_idx])
                    self.curriculum_levels.append(self.current_levels[env_idx])
                    self.min_distances.append(self.current_min_dists[env_idx])
                    self.target_distances.append(self.current_target_dists[env_idx])

                    # 重置当前回合数据
                    self.current_rewards[env_idx] = 0.0
                    self.current_lengths[env_idx] = 0
                    self.current_min_dists[env_idx] = float('inf')
                    self.current_target_dists[env_idx] = float('inf')

        return True

    def _on_rollout_end(self) -> None:
        # 在rollout结束时记录所有指标
        if self.episode_rewards:
            # 计算平均指标
            mean_reward = np.mean(self.episode_rewards)
            mean_length = np.mean(self.episode_lengths)
            mean_level = np.mean(self.curriculum_levels)
            mean_min_dist = np.mean(self.min_distances)
            mean_target_dist = np.mean(self.target_distances)

            # 记录到日志
            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("rollout/ep_len_mean", mean_length)
            self.logger.record("train/mean_curriculum_level", mean_level)
            self.logger.record("train/mean_min_distance", mean_min_dist)
            self.logger.record("train/mean_target_distance", mean_target_dist)

            # 重置存储
            self.episode_rewards = []
            self.episode_lengths = []
            self.curriculum_levels = []
            self.min_distances = []
            self.target_distances = []