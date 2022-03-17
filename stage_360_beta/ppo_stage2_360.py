import os
import logging
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from mpi4py import MPI

from torch.optim import Adam
from collections import deque

from sensor_msgs.msg import LaserScan
from collections import deque

import sys
sys.path.append("../")

from model_360.net import MLPPolicy, CNNPolicy
from stage_world2_360 import StageWorld
from model_360.ppo_360 import ppo_update_stage2, generate_train_data
from model_360.ppo_360 import generate_action, transform_buffer
from model_360.utils import get_group_terminal, get_filter_index

# train about
MAX_EPISODES = 5000
LASER_BEAM = 1080
LASER_HIST = 2
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 512
EPOCH = 4
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 44
OBS_SIZE = LASER_BEAM
ACT_SIZE = 2
LEARNING_RATE = 5e-5

# 360 about
FOV = 180  # 输入雷达的fov，单位是°
MAX_RANGE = 6.0  # 雷达归一化的最大值，单位是m
OUT_LINES = 1080   # 最终输出点个数


def run(comm, env, policy, policy_path, action_bound, optimizer, writer):
    buff = []
    global_update = 0
    global_step = 0

    if env.index == 0:
        env.reset_world()

    for id in range(MAX_EPISODES):
        env.reset_pose()

        env.generate_goal_point()
        group_terminal = False
        ep_reward = 0
        liveflag = True
        step = 1

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
        while not group_terminal and not rospy.is_shutdown():
            state_list = comm.gather(state, root=0)

            # generate actions at rank==0
            v, a, logprob, scaled_action = generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)
            
            # 执行动作
            real_action = comm.scatter(scaled_action, root=0)
            if liveflag == True:
                env.control_vel(real_action)
                # rate.sleep()
                rospy.sleep(0.001)
                # get informtion
                r, terminal, result = env.get_reward_and_terminate(step)
                step += 1


            if liveflag == True:
                ep_reward += r
            if terminal == True:
                liveflag = False

            global_step += 1

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

            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy,
                                                               action_bound=action_bound)
            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)

            terminal_list = comm.bcast(terminal_list, root=0)
            group_terminal = get_group_terminal(terminal_list, env.index)
            
            if env.index == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    filter_index = get_filter_index(d_batch)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    ppo_update_stage2(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory, filter_index=filter_index,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)

                    buff = []
                    global_update += 1

            state = state_next

        if env.index == 0:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage2_180_3060_{}.pth'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))

        if env.index == 40: #or env.index == 6 or env.index == 10 or env.index == 15 or env.index == 25 or env.index == 29 or env.index == 34 :
            reward_tag = 'robot_' + str(env.index) + '/ep_reward_episode'
            writer.add_scalar(reward_tag, scalar_value=ep_reward, global_step=id + 1)
            writer.flush()

            # writer.add_scalar('steps_episode', scalar_value=step, global_step=id + 1)
            # writer.flush()

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, %s,' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step-1, ep_reward, result))
        logger_cal.info(ep_reward)



if __name__ == '__main__':
    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    writer = SummaryWriter('./log/tensorboard/180_stage2_1.23')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = StageWorld(fov=FOV, max_range=MAX_RANGE, out_lines=OUT_LINES, beam_num=LASER_BEAM, index=rank, num_env=NUM_ENV)
    reward = None
    action_bound = [[0, -1], [1, 1]]  # [[v_min,w_min],[v_max,w_max]]
    
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'policy/180_stage2_1.23'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/Stage2_180_3060.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt, writer=writer)
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
