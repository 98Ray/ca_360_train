import os
import numpy as np
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from torch.optim import Adam
from collections import deque

from sensor_msgs.msg import LaserScan
import sys
sys.path.append("../")

from model_360.net import MLPPolicy, CNNPolicy
from world2_360_test import StageWorld
from model_360.ppo_360_single import generate_action_no_sampling, transform_buffer


MAX_EPISODES = 5000
LASER_BEAM = 1080
LASER_HIST = 2
HORIZON = 200
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 512
EPOCH = 3
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 1
OBS_SIZE = LASER_BEAM
ACT_SIZE = 2
LEARNING_RATE = 5e-5

# 360 about
FOV = 180  # 输入雷达的fov，单位是°
MAX_RANGE = 3.0  # 雷达归一化的最大值，单位是m
OUT_LINES = 1080   # 最终输出点个数



def enjoy(env, policy, action_bound):


    # env.reset_world()

    env.reset_pose()

    env.generate_goal_point()
    step = 1
    terminal = False
    result = None

    # 360 process
    lidar_filtered_topic = 'robot_' + str(env.index) + '/lidar_filtered'
    pub_filtered = rospy.Publisher(lidar_filtered_topic, LaserScan, queue_size=10)  # 发布消息到话题 chatter 中,队列长度10
    # pub_filtered = rospy.Publisher('lidar_filtered', LaserScan, queue_size=10)  # 发布消息到话题 chatter 中,队列长度10
    laser_filtered = LaserScan()

    lidar_polar = env.get_laser_polar()
    obs_his = deque([lidar_polar, lidar_polar, lidar_polar, lidar_polar])
    pose_his = deque([env.state_GT, env.state_GT, env.state_GT, env.state_GT])

    # TODO:①加入时间信息(bingo) ②改网络结构
    # lidar拼接
    r_filtered, t_feature = env.polars_full2r_filtered_rad(obs_his, pose_his)
    # rospy.loginfo("draw_lidar")  # 写入log日志
    laser_filtered.ranges = r_filtered.numpy().tolist()
    laser_filtered.header.frame_id = 'robot_' + str(env.index) + '/base_laser_link'
    laser_filtered.angle_min = 0
    laser_filtered.angle_max = 2 * np.pi
    laser_filtered.range_max = MAX_RANGE + 1
    laser_filtered.angle_increment = 2 * np.pi / OUT_LINES
    laser_filtered.intensities = t_feature.numpy().tolist()
    pub_filtered.publish(laser_filtered)

    # 生成归一化观测
    obs = r_filtered.numpy() / MAX_RANGE - 0.5
    obs_lidar_t = np.vstack((obs, t_feature.numpy()))  # 2*out_lines
    # print("obs_t.shape: ", obs_lidar_t.shape)
    obs_stack = deque([obs_lidar_t])
    goal = np.asarray(env.get_local_goal())  # [local_X,local_y]
    speed = np.asarray(env.get_self_speed())  # [linear.x,angular.z]
    state = [obs_stack, goal, speed]
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        # 测试一个循环时间大概6~20ms
        # import time
        # t0 = time.time()

        # # 执行之前获取新观测并看要不要更新
        # if env.update_his(pose_his[3]):
        #     obs_left = obs_his.popleft()
        #     obs_his.append(env.get_laser_polar())
        #     pose_left = pose_his.popleft()
        #     pose_his.append(env.state_GT)

        state_list = [state]

        # generate actions at rank==0
        mean, scaled_action =generate_action_no_sampling(env=env, state_list=state_list,
                                               policy=policy, action_bound=action_bound)


        # execute actions
        real_action = scaled_action[0]
        if terminal == True:
            if result == 'Crashed':
                pass
            else:
                real_action[0] = 0
                real_action[1] = 0
            print(result)
        else:
            real_action[0] = real_action[0]*0.2
            real_action[1] = real_action[1]*0.4
        # if terminal == True:
        #     if result == 'Reach Goal':
        #         print("Reach!")
        #         real_action[0] = 0
        #         real_action[1] = 0
        #         env.control_vel(real_action)

            # elif result == 'Time out':
            #     # print("Time out！")
            #     real_action[0] = 0
                
        env.control_vel(real_action)
        # rate.sleep()
        rospy.sleep(0.001)
        # get informtion
        r, terminal, result = env.get_reward_and_terminate(step)
        step += 1
        # 执行完之后获取新观测并看要不要更新
        if env.update_his(pose_his[3]):
            obs_left = obs_his.popleft()
            obs_his.append(env.get_laser_polar())
            pose_left = pose_his.popleft()
            pose_his.append(env.state_GT)

        # lidar拼接
        r_filtered, t_feature = env.polars_full2r_filtered_rad(obs_his, pose_his)
        # rospy.loginfo("draw_lidar")  # 写入log日志
        laser_filtered.ranges = r_filtered.numpy().tolist()
        laser_filtered.header.frame_id = 'robot_' + str(env.index) + '/base_laser_link'
        laser_filtered.angle_min = 0
        laser_filtered.angle_max = 2 * np.pi
        laser_filtered.range_max = MAX_RANGE + 1
        laser_filtered.angle_increment = 2 * np.pi / OUT_LINES
        laser_filtered.intensities = t_feature.numpy().tolist()
        pub_filtered.publish(laser_filtered)

        # 生成下一刻的归一化观测
        obs_next = r_filtered.numpy() / MAX_RANGE - 0.5
        obs_lidar_t_next = np.vstack((obs_next, t_feature.numpy()))
        obs_stack.popleft()
        obs_stack.append(obs_lidar_t_next)
        goal_next = np.asarray(env.get_local_goal())
        speed_next = np.asarray(env.get_self_speed())
        state_next = [obs_stack, goal_next, speed_next]


        state = state_next
        # t1 = time.time()
        # t_delta =int(round((t1-t0) * 1000))
        # print(t_delta)

        rate.sleep()




if __name__ == '__main__':

    rank = 0
    env = StageWorld(fov=FOV, max_range=MAX_RANGE, out_lines=OUT_LINES, beam_num=LASER_BEAM, index=7, num_env=NUM_ENV)
    reward = None
    action_bound = [[0, -1], [1, 1]]

    if rank == 0:
        policy_path = 'policy'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        # opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        # mse = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/Stage2_180_3060_420.pth'
        if os.path.exists(file):
            print ('####################################')
            print ('############Loading Model###########')
            print ('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            print ('Error: Policy File Cannot Find')
            exit()

    else:
        policy = None
        policy_path = None
        opt = None



    try:
        enjoy(env=env, policy=policy, action_bound=action_bound)
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
