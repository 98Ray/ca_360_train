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
from model_360.utils import test_init_pose, test_goal_point


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

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=10)

        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

        laser_topic = 'robot_' + str(index) + '/base_scan'

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        odom_topic = 'robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = 'robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)


        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)

        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None
        self.is_crashed = None
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None or self.is_crashed is None:
            pass


        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)


    def ground_truth_callback(self, GT_odometry):
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1


    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.data

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 6.0
        scan[np.isinf(scan)] = 6.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        return scan_sparse / 6.0 - 0.5


    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed

    def get_sim_time(self):
        return self.sim_time

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        rospy.sleep(0.5)


    def generate_goal_point(self):
        self.goal_point = test_goal_point(self.index)
        self.pre_distance = 0
        self.distance = copy.deepcopy(self.pre_distance)



    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5
        reward_c = 0
        reward_w = 0
        result = 0

        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        if is_crash == 1:
            terminate = True
            reward_c = -15.
            result = 'Crashed'

        if np.abs(w) >  0.7:
            reward_w = -0.1 * np.abs(w)

        if t > 10000:
            terminate = True
            result = 'Time out'
        reward = reward_g + reward_c + reward_w

        return reward, terminate, result

    def reset_pose(self):

        reset_pose = test_init_pose(self.index)
        self.control_pose(reset_pose)


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
        dis = math.sqrt(math.pow(xya_now[0] - xya_his[0], 2) + math.pow(xya_now[1] - xya_his[1], 2))  # cm
        theta = abs(xya_his[2] - xya_now[2])  # degree
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
        obs_his_to_now = u_c.hisPolars2nowPolar_rad(obs_his, poses_his, self.state_GT)
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
