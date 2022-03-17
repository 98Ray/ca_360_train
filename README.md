## 训练说明
### 仿真环境准备
参考https://github.com/Acmece/rl-collision-avoidance的仿真环境构建  

### 文件描述
1. policy中是训练的模型文件  
2. 360_visualize_stage_better.py用于360°的可视化效果  
3. stage2_test_single.py用于在stage中测试模型效果，使用的世界是stage_world2_360_test.py  
4. ppo_stage1_360和ppo_stage2_360分别用于两个阶段的训练
5. transformations.py是ros的库扒下来的  
6. key_stage1.py可用于键盘控制及获取坐标点   
7. util_coordinate包含用于坐标转换的函数，util_coordinate_simplify是在部署的时候把用到的函数删减出来的，较为精炼些
8. rviz中是stage的可视化工具rviz的配置文件

### 训练步骤：  
  `rosrun stage_ros_add_pose_and_crash stageros -g worlds/stage1.world`   
  `mpiexec -np 24 python ppo_stage1_360.py`  
  `rosrun stage_ros_add_pose_and_crash stageros -g worlds/stage2.world`  
  `mpiexec -np 44 python ppo_stage2_360.py`   