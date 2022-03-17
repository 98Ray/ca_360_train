## 训练说明
- ppo_stage1_360和ppo_stage2_360分别用于两个阶段的训练  
- rviz中是stage的可视化工具rviz的配置文件     
- 训练步骤：  
  `rosrun stage_ros_add_pose_and_crash stageros -g worlds/stage1.world`   
  `mpiexec -np 24 python ppo_stage1_360.py`  
  `rosrun stage_ros_add_pose_and_crash stageros -g worlds/stage2.world`  
  `mpiexec -np 44 python ppo_stage2_360.py`  