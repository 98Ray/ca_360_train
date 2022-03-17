# coding=utf-8
import rospy
import numpy as np
from stage_world1_360_visualize import StageWorld
from sensor_msgs.msg import LaserScan
from collections import deque

FOV = 150  # 单位是°
MAX_RANGE = 6.0  # 单位是mm
OUT_LINES = 360   # 个数
LASER_BEAM = 512

def run(env):
    pub_filtered = rospy.Publisher('lidar_filtered', LaserScan, queue_size=10)  # 发布消息到话题 chatter 中,队列长度10
    laser_filtered = LaserScan()

    lidar_polar = env.get_laser_polar()
    obs_his = deque([lidar_polar, lidar_polar, lidar_polar, lidar_polar])
    pose_his = deque([env.state_GT, env.state_GT, env.state_GT, env.state_GT])
    rate = rospy.Rate(1)  # 10hz  设置发布频率

    while not rospy.is_shutdown():
        # lidar拼接
        r_filtered = env.polars_full2r_filtered_rad(obs_his, pose_his)
        # rospy.loginfo("draw_lidar")  # 写入log日志
        laser_filtered.ranges = r_filtered.numpy().tolist()
        laser_filtered.header.frame_id = "/robot_0/base_laser_link"
        laser_filtered.angle_min = 0
        laser_filtered.angle_max = 2 * np.pi
        laser_filtered.range_max = MAX_RANGE + 0.1
        laser_filtered.angle_increment = 2 * np.pi / OUT_LINES
        pub_filtered.publish(laser_filtered)

        # 执行动作
        # env.control_pose([env.state_GT[0], env.state_GT[1], env.state_GT[2]+15/180*np.pi])
        # env.control_vel([0.6, 0.4])

        # 执行完之后获取新观测并看要不要更新
        if env.update_his(pose_his[3]):
            obs_left = obs_his.popleft()
            obs_his.append(env.get_laser_polar())
            pose_left = pose_his.popleft()
            pose_his.append(env.state_GT)
        # rate.sleep()  # 配合发布频率的休眠

if __name__ == '__main__':
    env = StageWorld(fov=FOV, max_range=MAX_RANGE, out_lines=OUT_LINES, beam_num=LASER_BEAM, index=0)
    try:
        run(env=env)
    except KeyboardInterrupt:
        pass
