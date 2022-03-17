# coding=utf-8
import time
import rospy
import copy
import transformations
import numpy as np
import utils_coordinate as u_c
import torch
import math

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class StageWorld():
    def __init__(self, fov, max_range, out_lines, beam_num, index, num_env=1):
        self.index = index  # no use
        self.num_env = num_env  # no use
        node_name = 'StageEnv_' + str(index)  # rl no use
        rospy.init_node(node_name, anonymous=None)  # rl no use

        self.beam_mum = beam_num
        self.laser_cb_num = 0  # no use
        self.scan = None

        self.lidar_polar = None  # N*2的array
        self.fov = fov
        self.max_range = max_range
        self.out_lines = out_lines

        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m  #no use
        self.goal_size = 0.5  # 根据距离是否小于这个值判断是否到达目的地

        self.robot_value = 10.  # no use
        self.goal_value = 0.   # no use
        # self.reset_pose = None

        self.now_pose = None

        # -----------Publisher and Subscriber-------------
        # 为机器人发布指定速度（线速度+加速度）
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        # 主要目的就是为了得到机器人当前的线速度和角速度
        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

        laser_topic = 'robot_' + str(index) + '/base_scan'
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)

        # # Wait until the first callback
        self.state_GT = None  #v,v_angular.z # no use
        while self.scan is None or self.state_GT is None:
            pass

        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)

    # no use
    # 获取gt的v和a，但是并没有被用到
    def ground_truth_callback(self, GT_odometry):
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]

    # 获取当前雷达数据
    def laser_scan_callback(self, scan):
        self.scan = np.array(scan.ranges)

    # no use
    def get_self_stateGT(self):
        return self.state_GT

    # 控制速度
    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)

    def control_pose(self, pose):
        pose_cmd = Pose()
        assert len(pose)==3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)

    def get_laser_polar(self):
        lidar_data = copy.deepcopy(self.scan)
        lidar_data[np.isnan(lidar_data)] = self.max_range  # 是否为空  stage中的机器人的range是[0.0,6.0],fov=180,num是512
        lidar_data[np.isinf(lidar_data)] = self.max_range  # 是否无界
        lidar_data[lidar_data > self.max_range] = self.max_range
        lidar_polar = u_c.range2polar(lidar_data, self.fov/180*np.pi)
        return lidar_polar

    def update_his(self, xya_his):
        # xya_his是列表，含有三个元素xya
        xya_now = self.state_GT
        # 返回是否需要更新历史信息
        dis = math.sqrt(math.pow(xya_now[0] - xya_his[0], 2) + math.pow(xya_now[1] - xya_his[1], 2))  # cm
        theta = abs(xya_his[2] - xya_now[2])  # degree
        # print("now_a:{},his_a{}:".format(xya_now[2], xya_his[2]))
        # print("now_xy:({},{}),his_xy:({},{})".format(xya_now[0],xya_now[1],xya_his[0],xya_his[1]))
        # 距离大于20cm或者角度大于10°
        if dis > 0.2 or theta > 0.175:
            # print("theta:{},dis{}:".format(theta, dis))
            print("Update!")
            return True
        else:
            return False

    def polars_full2r_filtered_rad(self, obs_his, poses_his):
        obs_his_to_now = u_c.hisPolars2nowPolar_rad(obs_his, poses_his, self.state_GT)
        lidar_polar = self.get_laser_polar()
        obs_full_polar = torch.cat((obs_his_to_now, lidar_polar.unsqueeze(0)), dim=0)
        full_r_id = u_c.polars2index(input_rps=obs_full_polar, out_lines=self.out_lines, scope_keep=2 * np.pi)  # his*N*2
        r_filtered, t_feature = u_c.index_filter(full_r_id, self.max_range, self.out_lines)

        return r_filtered, t_feature


if __name__ == '__main__':
    from collections import deque
    obs_his = deque([torch.tensor([[1, -np.pi]])])
    poses_his = deque([[3, 1, 0]])
    pose_now = [4, 1, 0]
    obs_his_to_now = u_c.hisPolars2nowPolar_rad(obs_his, poses_his, pose_now)
    print("wow")
