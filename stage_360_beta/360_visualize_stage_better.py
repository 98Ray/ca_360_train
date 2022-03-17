# coding=utf-8
import rospy
import numpy as np
from world_360_visualize import StageWorld
from sensor_msgs.msg import LaserScan
from collections import deque
from geometry_msgs.msg import Twist

FOV = 180  # 单位是°
MAX_RANGE = 6.0  # 单位是m
OUT_LINES = 1080   # 个数
LASER_BEAM = 1080
vel = [0, 0]

def callback(msg):
    # rospy.loginfo("I heard %s", msg.data)
    print("in callback")
    global vel
    vel[0] = msg.linear.x
    vel[1] = msg.angular.z

def run(env):
    pub_filtered = rospy.Publisher('/robot_7/lidar_filtered', LaserScan, queue_size=10)  # 发布消息到话题 chatter 中,队列长度10
    rospy.Subscriber("/robot_7/cr_vel", Twist, callback)
    laser_filtered = LaserScan()

    lidar_polar = env.get_laser_polar()
    obs_his = deque([lidar_polar, lidar_polar, lidar_polar, lidar_polar])
    pose_his = deque([env.state_GT, env.state_GT, env.state_GT, env.state_GT])
    # rate = rospy.Rate(1)  # 10hz  设置发布频率

    # lidar拼接
    r_filtered, t_feature = env.polars_full2r_filtered_rad(obs_his, pose_his)
    # rospy.loginfo("draw_lidar")  # 写入log日志
    laser_filtered.ranges = r_filtered.numpy().tolist()
    laser_filtered.header.frame_id = "/robot_7/base_laser_link"
    laser_filtered.angle_min = 0
    laser_filtered.angle_max = 2 * np.pi
    laser_filtered.range_max = MAX_RANGE + 1
    laser_filtered.angle_increment = 2 * np.pi / OUT_LINES
    laser_filtered.intensities = t_feature.numpy().tolist()
    pub_filtered.publish(laser_filtered)
    rate = rospy.Rate(8)

    while not rospy.is_shutdown():
        # import time
        # t0 = time.time()
        # 执行之前获取新观测并看要不要更新
        if env.update_his(pose_his[3]):
            obs_left = obs_his.popleft()
            obs_his.append(env.get_laser_polar())
            pose_left = pose_his.popleft()
            pose_his.append(env.state_GT)

        # 执行动作
        # env.control_pose([env.state_GT[0], env.state_GT[1], env.state_GT[2]+10/180*np.pi])
        global vel
        env.control_vel(vel)
        # # 执行完之后获取新观测并看要不要更新
        # if env.update_his(pose_his[3]):
        #     obs_left = obs_his.popleft()
        #     obs_his.append(env.get_laser_polar())
        #     pose_left = pose_his.popleft()
        #     pose_his.append(env.state_GT)

        # lidar拼接
        r_filtered, t_feature = env.polars_full2r_filtered_rad(obs_his, pose_his)
        # rospy.loginfo("draw_lidar")  # 写入log日志
        laser_filtered.ranges = r_filtered.numpy().tolist()
        laser_filtered.header.frame_id = "/robot_7/base_laser_link"
        laser_filtered.angle_min = 0
        laser_filtered.angle_max = 2 * np.pi
        laser_filtered.range_max = MAX_RANGE + 1
        laser_filtered.angle_increment = 2 * np.pi / OUT_LINES
        laser_filtered.intensities = t_feature.numpy().tolist()
        pub_filtered.publish(laser_filtered)
        # rate.sleep()
        # t1 = time.time()
        # t_delta =int(round((t1-t0) * 1000))
        # print(t_delta)  # 2ms



if __name__ == '__main__':
    env = StageWorld(fov=FOV, max_range=MAX_RANGE, out_lines=OUT_LINES, beam_num=LASER_BEAM, index=7)
    try:
        run(env=env)
    except KeyboardInterrupt:
        pass
