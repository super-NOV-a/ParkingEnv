这里提供了泊车环境的基础功能，包括gym接口、渲染功能、环境生成与处理等。该文件结构如下：

parking_env_pkg/
│
├── parking_core.py          # ParkingEnv 主循环需要的：奖励、终止等gym接口
├── render.py                # 使用 pygame 对场景（自车、车位和障碍）进行绘制
├── scenario_manager.py      # 加载 / 随机生成场景  ---  后续需要课程学习在此处进行修改！
└── utils.py                 # 角度归一化、车与车位角顶点计算
