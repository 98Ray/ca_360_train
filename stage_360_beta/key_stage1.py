# !/usr/bin/env python
# -*- coding:utf-8   -*-

import sys
import select
import tty
import termios
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import transformations

state_GT = None

def ground_truth_callback(GT_odometry):
    Quaternious = GT_odometry.pose.pose.orientation
    Euler = transformations.euler_from_quaternion([Quaternious.x,Quaternious.y, Quaternious.z, Quaternious.w])
    global state_GT
    state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
    # print("in callback")


if __name__ == '__main__':
    key_pub = rospy.Publisher('robot_7/cmd_vel', Twist, queue_size=1)
    rospy.init_node('keyboard_driver')
    object_state_topic = 'robot_' + str(7) + '/base_pose_ground_truth'
    object_state_sub = rospy.Subscriber(object_state_topic, Odometry, ground_truth_callback)
    rate = rospy.Rate(10)
    twist = Twist()

    # 保存原来属性
    old_attr = termios.tcgetattr(sys.stdin)
    # 设置为单字符响应模式
    tty.setcbreak(sys.stdin.fileno())
    print("Publishing keystrokes. Press Ctrl+C to exit...")
    while not rospy.is_shutdown():
        if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
            # 发布按键
            word = sys.stdin.read(1)
            if word == 'w':
                twist.linear.x += 0.05
                print("w")
            elif word == 'x':
                twist.linear.x -= 0.05
            elif word == 'a':
                twist.angular.z += 0.05
                # twist.angular.z = 0.8
            elif word == 'd':
                twist.angular.z -= 0.05
                # twist.angular.z = -0.8
            elif word == 's':
                twist.angular.z = 0
                twist.linear.x = 0
            elif word == '1':
                print("1")
            elif word == 'p':
                print("State: ", state_GT)
            print("Now speed:[{} cm/s],angular:[{} degree/s]".format(str(twist.linear.x), str(twist.angular.z)))
            key_pub.publish(twist)
        rate.sleep()
    # 恢复属性
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)