# coding=utf-8
import time
import rospy
import copy
import transformations
import numpy as np
import sys
sys.path.append("../")
import utils_coordinate as u_c
import torch
import math

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8
from model_360.utils import get_init_pose, get_goal_point


class StageWorld():
    def __init__(self, fov, max_range, out_lines, beam_num, index, num_env):
        self.index = index
        self.num_env = num_env
        node_name = 'StageEnv_' + str(index)
        rospy.init_node(node_name, anonymous=None)

        self.beam_mum = beam_num
        self.laser_cb_num = 0
        self.scan = None

        # my new added
        self.lidar_polar = None  # N*2的array
        self.fov = fov
        self.max_range = max_range
        self.out_lines = out_lines

        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = 0.5

        self.robot_value = 10.
        self.goal_value = 0.
        # self.reset_pose = None

        self.init_pose = None  #no use

        # for get reward and terminate
        self.stop_counter = 0  # no use

        # -----------Publisher and Subscriber-------------
        # 为机器人发布指定速度（线速度+加速度）
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        # 将机器人设置到指定的位姿（位置+转角）
        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=10)

        # 主要目的就是为了得到机器人当前的线速度和角速度
        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

        laser_topic = 'robot_' + str(index) + '/base_scan'
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        # 主要目的就是为了从里程计信息中获取机器人当前的速度
        odom_topic = 'robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = 'robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)

        # self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)  # no use

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)  # ginger_sim no use

        # # Wait until the first callback
        self.speed = None
        # self.state = None
        self.speed_GT = None
        self.state_GT = None  #v,v_angular.z # my use
        # self.is_crashed = None
        while self.scan is None or self.speed is None \
                or self.speed_GT is None or self.state_GT is None:
            pass

        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)

    # my use
    # 本来没用，但被我用来获取当前位姿了
    def ground_truth_callback(self, GT_odometry):
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x ** 2 + v_y ** 2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    # 获取当前雷达数据
    def laser_scan_callback(self, scan):
        self.scan = np.array(scan.ranges)

    # 获取机器人当前速度
    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = transformations.euler_from_quaternion(
            [Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    # 是否碰撞
    def crash_callback(self, flag):
        self.is_crashed = flag.data
        # print("is_crashed: ", self.is_crashed)

    def get_self_speedGT(self):
        return self.speed_GT

    # my use
    def get_self_stateGT(self):
        return self.state_GT

    # 获取速度
    def get_self_speed(self):
        return self.speed

    # 获取是否碰撞
    def get_crash_state(self):
        return self.is_crashed

    # 获取目标点在机器人坐标系下的坐标
    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    # 重置世界，只是为了在第一个进程中重置世界，对于ginger仿真可有可无
    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        rospy.sleep(0.5)

    # 生成随机目标点
    def generate_goal_point(self):
        if self.index > 33 and self.index < 44:
            self.goal_point = self.generate_random_goal()
        else:
            self.goal_point = get_goal_point(self.index)

        self.pre_distance = 0
        self.distance = copy.deepcopy(self.pre_distance)

    # 获取奖励与是否达到终止条件
    def get_reward_and_terminate(self, t):
        terminate = False
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5  # w_g=2.5
        reward_c = 0
        reward_w = 0
        result = 0
        is_crash = self.get_crash_state()

        # 到达目的地
        if self.distance < self.goal_size:  # goal_size=0.5
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        # 发生碰撞
        if is_crash == 1:
            terminate = True
            reward_c = -15.
            result = 'Crashed'

        # 角速度大于阈值给一个惩罚
        # ω_w=-0.1
        if np.abs(w) > 1.05:
            reward_w = -0.1 * np.abs(w)

        # 超时停止,step>200
        # stage1中是150
        if t > 200:
            terminate = True
            result = 'Time out'
        reward = reward_g + reward_c + reward_w

        return reward, terminate, result

    # 重置机器人的位姿
    def reset_pose(self):
        if self.index > 33 and self.index < 44:
            reset_pose = self.generate_random_pose()
        else:
            reset_pose = get_init_pose(self.index)
        rospy.sleep(0.05)
        self.control_pose(reset_pose)
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        while np.abs(reset_pose[0] - x_robot) > 0.2 or np.abs(reset_pose[1] - y_robot) > 0.2:
            [x_robot, y_robot, theta] = self.get_self_stateGT()
        rospy.sleep(0.05)

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


    def generate_random_pose(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(9, 19)
        y = np.random.uniform(0, 1)
        if y <= 0.4:
            y = -(y * 10 + 1)
        else:
            y = -(y * 10 + 9)
        dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        while (dis_goal < 7) and not rospy.is_shutdown():
            x = np.random.uniform(9, 19)
            y = np.random.uniform(0, 1)
            if y <= 0.4:
                y = -(y * 10 + 1)
            else:
                y = -(y * 10 + 9)
            dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        theta = np.random.uniform(0, 2*np.pi)
        return [x, y, theta]


    def generate_random_goal(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(9, 19)
        y = np.random.uniform(0, 1)
        if y <= 0.4:
            y = -(y*10 + 1)
        else:
            y = -(y*10 + 9)
        dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        while (dis_goal < 7) and not rospy.is_shutdown():
            x = np.random.uniform(9, 19)
            y = np.random.uniform(0, 1)
            if y <= 0.4:
                y = -(y * 10 + 1)
            else:
                y = -(y * 10 + 9)
            dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        return [x, y]

####### My functions########
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
        dis = math.sqrt(math.pow(xya_now[0] - xya_his[0], 2) + math.pow(xya_now[1] - xya_his[1], 2))  # m
        theta = abs(xya_his[2] - xya_now[2])  # rad
        # print("now_a:{},his_a{}:".format(xya_now[2], xya_his[2]))
        # print("now_xy:({},{}),his_xy:({},{})".format(xya_now[0],xya_now[1],xya_his[0],xya_his[1]))
        # 距离大于20cm或者角度大于10°
        if dis > 0.2 or theta > 0.175:
            # print("theta:{},dis{}:".format(theta, dis))
            # print("Update!")
            return True
        else:
            return False

    def polars_full2r_filtered_rad(self, obs_his, poses_his):
        obs_his_to_now = u_c.hisPolars2nowPolar_rad(obs_his, poses_his, self.state_GT, self.max_range)
        lidar_polar = self.get_laser_polar()
        obs_full_polar = torch.cat((obs_his_to_now, lidar_polar.unsqueeze(0)), dim=0)
        # full_r_id : his*N*2
        full_r_id = u_c.polars2index(input_rps=obs_full_polar, out_lines=self.out_lines, scope_keep=2 * np.pi)  # his*N*2
        r_filtered, t_feature = u_c.index_filter(full_r_id, self.max_range, self.out_lines)

        return r_filtered, t_feature


if __name__ == '__main__':
    from collections import deque
    obs_his = deque([torch.tensor([[1, -np.pi]])])
    poses_his = deque([[3, 1, 0]])
    pose_now = [2, 2, 5*np.pi/4]
    obs_his_to_now = u_c.hisPolars2nowPolar(obs_his, poses_his, pose_now)
    print("wow")
