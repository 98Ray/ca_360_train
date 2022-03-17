# coding=utf-8
# !/home/joeray/Programs/miniconda3/envs/one_lidar_rl/bin python

import torch
from torch_scatter import scatter_min
import numpy as np
import math

# used
def index_filter(n_r_id, max_range, out_lines):
    # n_r_id是his*N*2的tensor，也就是要把所有观测放在一个tensor中再送进来；max_range是雷达的最大值；out_lines是最终输出的线数
    tmp = torch.transpose(n_r_id, 1, 2)  # 转成r一行，id一行 his*2*N
    new_r_id = None
    # 把所有的r拼成一行，id拼成一行
    # 最终拼完的new_r_id 形状：2*(his*N),一行是r，一行是id，每行有his*N个数据
    for i in range(len(tmp)):
        if i == 0:
            new_r_id = tmp[0]
        else:
            new_r_id = torch.cat((new_r_id, tmp[i]), dim=1)

    # 关于时间的一维特征信息，因为从前到后是t越来越靠近当前，所以时间特征越来越大，以0.2倍增
    # TODO:以下一部分操作和输入的雷达线数有关，如果不是512的话需要改一下
    t_feature = torch.empty([2560])
    for i in range(5):
        t_feature[i*512:(i+1)*512] = 0.2*(i+1)

    # 返回的均为OUT_LINES长度的tensor
    # 获取到对应角度的最小值以及最小值对应的序号
    filtered_r, argmin = scatter_min(src=new_r_id[0], index=new_r_id[1].to(torch.long))
    argmin[argmin == 2560] = 2559  # 不知为何会出现2560的index，超出了src的长度
    t_feature_new = t_feature[argmin]  # 将对应距离最小值的t特征取出
    id_unknown = torch.where(filtered_r == 0)  #得到filtered_r中未知区域的序号
    t_feature_new[id_unknown] = -1  # 将未知区域的t特征设置为-1
    filtered_r[filtered_r == 0] = max_range  # 历史观测未探测的未知区域的距离特征设置为最大值

    # 扩充原始的当前帧雷达数据
    # 扩充到1080的一半，为了让眼前的这180度都是确定的当前帧（如果改成150，那么需要把1/2改成150/360）
    r_now = new_r_id[0][4*512:]
    raw_beam_num = 512
    sparse_beam_num = out_lines/2
    step = float(raw_beam_num) / sparse_beam_num
    sparse_scan_left = []
    index = 0.
    for x in range(int(sparse_beam_num / 2)):
        sparse_scan_left.append(r_now[int(index)])
        index += step
    sparse_scan_right = []
    index = raw_beam_num - 1.
    for x in range(int(sparse_beam_num / 2)):
        sparse_scan_right.append(r_now[int(index)])
        index -= step
    scan_sparse = torch.from_numpy(np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0))

    # 用当前帧替换掉filtered_r中的当前观测部分
    id_near2zero = int(180/360*out_lines/2)
    id_near2last = int(out_lines - 180/360*out_lines/2)
    filtered_r[:id_near2zero] = scan_sparse[int(out_lines/4):]
    filtered_r[id_near2last:] = scan_sparse[:int(out_lines / 4)]
    # 将当前帧对应的时间特征全部改为1.0
    t_feature_new[:id_near2zero] = 1.0
    t_feature_new[id_near2last:] = 1.0


    return filtered_r, t_feature_new

# used ok
def polar2index(input_rp, out_lines, scope_keep=math.pi * 2):
    # 假设input_rp依然是N*2的tensor
    # 返回值是N*2的tensor，代表r_id对
    # 将角度统一转到大于零的范围，避免因角度为负值造成的index赋值不正确的现象。（因为之后的int操作会向靠近0的方向取整）
    input_rp[input_rp < 0] += scope_keep
    step = scope_keep / out_lines
    output_r = input_rp[:, 0]
    # input_rp[input_rp[:, 1] < 0]
    output_id = (input_rp[:, 1] / step).to(torch.int64)
    output_r_id = torch.stack((output_r, output_id), dim=1)
    return output_r_id

# used ok
def polars2index(input_rps, out_lines, scope_keep=math.pi * 2):
    # 假设input_rps是his*N*2的tensor
    # 返回值是his*N*2的tensor，代表r_id对
    output_r_ids = []
    for i in range(len(input_rps)):
        output_r_ids.append(polar2index(input_rps[i], out_lines, scope_keep))
    return torch.stack(output_r_ids)

# used ok
def hisPolar2nowPolar_rad(rp_his, xya_his, xya_now, max_range):
    # 输入的a是rad，输出的p是rad
    # rp_his是N*2的tensor，xya_his和xya_now都是列表，含有三个元素xya
    # 输出是N*2的tensor
    xy_his = polar2cart_rad(rp_his)
    xy_now = his_xy2now_xy(xya_his, xya_now, xy_his)
    # matrix = getTransMatrix_rad(xya_his, xya_now)
    # xy_now = getPosInNow(xy_his, matrix)
    rp_now = cart2polar(xy_now, max_range)
    # 因为雷达未扫到的点被初始化为max_range
    # 然后当机器人向前移动且依然没有被遮挡时，会出现未被遮挡处的点逐渐靠近的情况
    # 以下的操作是为了将初始化为max_range（也就是补的max_range值）不做变换
    # 以此来避免上述情况的出现
    r_tmp = torch.where(rp_his[:, 0] == max_range, rp_his[:, 0], rp_now[:, 0])
    rp_now.transpose(0, 1)[0] = r_tmp
    return rp_now


# used ok
def hisPolars2nowPolar_rad(rps_his, xyas_his, xya_now, max_range, update):
    # 输入的a是rad，输出的p是rad
    # rps_his是his*N*2的tensor，xyas_his是his*3的列表，xya_now含有三个元素xya
    # 输出是his*N*2的tensor
    if update:
        print("delta_deg_his2now", np.degrees(xya_now[2] - np.array(xyas_his)[:, 2]))  # 当前角度减历史角度,复制
    rps_now = []
    for i in range(len(rps_his)):
        rps_now.append(hisPolar2nowPolar_rad(rps_his[i], xyas_his[i], xya_now, max_range))
    return torch.stack(rps_now, dim=0)


# used ok
def range2polar(ranges, fov):
    # ranges为np.ndarray类型，fov是一个值
    # 返回N*2的tensor代表(r,p)对，从-fov/2开始到fov/2
    num_lines = len(ranges)
    resolution = fov / num_lines
    ranges_theta = torch.from_numpy(np.arange(-fov / 2.0 + resolution / 2.0, fov / 2.0, resolution))
    ranges = torch.from_numpy(ranges)
    ranges_polar = torch.stack((ranges, ranges_theta), dim=1)
    return ranges_polar


# used ok
# 左手系
# 返回值的大小是从-π到π的
def cart2polar(input_xy, max_range):
    # input_xy是N*2的tensor
    # 返回N*2大小tensor的(r,theta)对
    r = torch.sqrt(torch.pow(input_xy[:, 0], 2) + torch.pow(input_xy[:, 1], 2))
    # 因为在之前的坐标转换过程中，也就是极坐标到直角坐标，历史直角坐标再到当前直角坐标的过程中会有一些偏差
    # 所以会导致转换得到的最终极坐标系下的距离值超过max_range，虽然超出的值不大也就0.1、0.2左右，但还是要加一个限幅比较好
    r[r > max_range] = max_range
    p = torch.atan2(input_xy[:, 1], input_xy[:, 0])
    return torch.stack((r, p), dim=1)

# used  ok
def polar2cart_rad(input_rp):
    # input_rp是N*2的tensor
    # 返回N*2大小tensor的(x,y)坐标对
    cos = torch.cos(input_rp[:, 1])
    sin = torch.sin(input_rp[:, 1])
    x = torch.mul(input_rp[:, 0], cos)
    y = torch.mul(input_rp[:, 0], sin)
    return torch.stack((x, y), dim=1)


# used ok
def his_xy2now_xy(xya_his, xya_now, xy_his):
    # xya_his和xya_now都是list
    # xy_his是N*2的tensor
    his_m = pose2matrix(np.array(xya_his))  # 3*3 matrix
    now_m = pose2matrix(np.array(xya_now))  # 3*3 matrix
    xy_his_homo = np.concatenate((xy_his.numpy().T, np.ones((1, xy_his.shape[0]))), 0)  # 3*N
    xy_now = (np.linalg.inv(now_m) @ his_m @ xy_his_homo)[:2, :].T  # N*2
    return torch.from_numpy(xy_now)

# used ok
def pose2matrix(pose: np.ndarray):
    theta_rad = pose[2]
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    t_x = pose[0]
    t_y = pose[1]
    matrix = np.array([[cos_theta, -sin_theta, t_x],
                       [sin_theta, cos_theta, t_y],
                       [0, 0, 1]])
    return matrix


if __name__ == '__main__':
    # # '''测试'''
    # # print(pose2matrix(np.array([2, 1, np.pi/4])))
    # xya_his = [2, 1, np.pi/4]
    # xya_now = [1, 1, np.pi/2]
    # xy_his = torch.tensor([
    #     [0.707, 0.707],
    #     [0.707, -0.707]
    # ])
    # print(his_xy2now_xy(xya_his, xya_now, xy_his))

    '''测试用rviz可视化turtlebot补全后的雷达点'''
    # stage_subscriber()

    '''测试用rviz实时可视化雷达点'''
    # with grpc.insecure_channel('10.21.255.107:30001',options=[
    #         ('grpc.max_send_message_length', 1024*1024*1024),
    #         ('grpc.max_receive_message_length', 1024*1024*1024)
    #     ]) as channel:
    #     stub = GrabSim_pb2_grpc.GrabSimStub(channel)
    #     rospy.init_node('talker_lidar', anonymous=True)  # 初始化节点名字为talker,加入一个随机数使得节点名称唯一
    #     rate = rospy.Rate(2)  # 10hz  设置发布频率
    #     pub = rospy.Publisher('chatter', LaserScan, queue_size=10)  # 发布消息到话题 chatter 中,队列长度10
    #     while not rospy.is_shutdown():  # 当没有异常关闭时候执行如下程序(防止ctrl+c 终止程序)
    #         lidar = np.array(lidar_collect(stub))
    #         # print(lidar)
    #         lidar_polar = range2polar(lidar, 217.8 / 180 * np.pi)  # 角度有正有负 array (242,)
    #         # print(lidar_polar)
    #         lidar_r_id = polar2index(input_rp=lidar_polar, out_lines=180, scope_keep=2 * np.pi)
    #         lidar_r_id = lidar_r_id.unsqueeze(0)
    #         # print(lidar_r_id)
    #         r_filtered = index_filter(lidar_r_id, max_range=1000, out_lines=180)
    #
    #         r_filtered = r_filtered/1000  #转到m的尺度下
    #         # 倒序，为了和ros的坐标系匹配
    #         r_filtered = r_filtered.numpy()[::-1]
    #         r_filtered = torch.from_numpy(r_filtered.copy())
    #
    #         rospy.loginfo("draw_lidar")  # 写入log日志
    #         laser_filtered = LaserScan()
    #         laser_filtered.ranges = r_filtered.numpy().tolist()
    #         laser_filtered.header.frame_id = "lidar"
    #         laser_filtered.angle_min = 0
    #         laser_filtered.angle_max = 2 * np.pi
    #         laser_filtered.range_max = 1000
    #         laser_filtered.angle_increment = 2 * np.pi / 180
    #         pub.publish(laser_filtered)  # 发布字符串
    #         rate.sleep()  # 配合发布频率的休眠

    '''测试从polar2index到最终的一维tensor的全过程'''
    '''假设一圈以4度代表，最终以8个点代表这4度中的值，每个点代表0.5度的范围'''
    '''每次观测fov是2度，lines是3个'''
    # input_rp_1 = torch.tensor([[0.1, -0.25], [0.1, -0.75], [0.2, 1.25]])
    # input_rp_2 = torch.tensor([[0.05, 0.75], [0.05, 0.75], [0.2, 1.75]])
    # input_rp_3 = torch.tensor([[0.1, 1.75], [0.2, 2.75], [0.2, 2.75]])
    # input_rps = torch.stack((input_rp_1, input_rp_2, input_rp_3), dim=0)
    # out_r_ids = polars2index(input_rps, 8, 4)
    # filtered_points = index_filter(out_r_ids, max_range=0.4, out_lines=8)
    # print(filtered_points)
    # draw_fullOb(filtered_points, full_angle=4)

    '''测试r_id连接并在对应id找最小值'''
    # n_r_id = torch.tensor([
    #                         [[1, 45], [2, 45], [3, 0]],
    #                         [[1, 90], [2, 90], [2, 0]],
    #                         [[3, 45], [1, 135], [4, 90]]
    #                       ])
    # print(index_filter(n_r_id))

    '''测试极坐标系转对应序号'''
    # input_rp = torch.tensor([[1.414, -math.pi/4], [1, -math.pi/4], [2, math.pi/2], [1, 0]])
    # print(polar2index(input_rp, num_lines=360))

    '''测试历史极坐标系转当前极坐标系(这个有两种，一个是deg的，一个是rad的)'''
    # rp_his = torch.tensor([[1, 180], [1.414, -135]], dtype=torch.float)
    # xya_his = [3, 1, 0]
    # xya_now = [2, 2, 225]
    # print(hisPolar2nowPolar_deg(rp_his, xya_his, xya_now))

    # rp_his = torch.tensor([[1, 180], [1.414, -135]], dtype=torch.float)
    # xya_his = [3, 1, 0]
    # xya_now = [2, 2, 225]
    # print(hisPolar2nowPolar_rad(rp_his, xya_his, xya_now))

    '''测试更新历史'''
    # print(update_his([3, 1, 0], [2, 2, -45]))

    '''测试雷达点转极坐标系'''
    # print(range2polar(np.ones((115)), 265))

    '''测试笛卡尔坐标系与极坐标系之间的相互转换
    # 37° = 0.6458
    # 45° = 0.7854
    # 53° = 0.9273
    # 90° = 1.5708
    '''
    # data_cart = [[3, 4],
    #              [4, 3]]
    # #              [-3, 4],
    # #              [-4, 3],
    # #              [-4, -3],
    # #              [3, -4]]
    # data_polar = cart2polar(torch.FloatTensor(data_cart), 5.0)
    # print("Polar:", data_polar)
    # print("Cart:", polar2cart_rad(data_polar))

    '''测试变换矩阵以及求历史坐标在当前坐标系下的坐标'''
    # xya_his = [3, 1, 0]
    # xya_now = [2, 2, -225]
    # matrix = getTransMatrix(xya_his, xya_now)
    # print(matrix)
    # input_xy = torch.FloatTensor([[-1, -1], [-1, 0], [-2, 0]])
    # print getPosInNow(input_xy, matrix)

    # 测试左手系的坐标转换正确性
    xya_his = [2, 1, 0]
    xya_now = [1, 1, np.pi/2]
    xy_his = torch.tensor([
        [1, 1],
        [0, 0]
    ])
    print(his_xy2now_xy(xya_his, xya_now, xy_his))