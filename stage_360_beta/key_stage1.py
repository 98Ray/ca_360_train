# !/usr/bin/env python
# -*- coding:utf-8   -*-

import sys
import select
import tty
import termios
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist




if __name__ == '__main__':
    key_pub = rospy.Publisher('robot_0/cmd_vel', Twist, queue_size=1)
    rospy.init_node('keyboard_driver')
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
                twist.linear.x += 0.1
                print("w")
            elif word == 'x':
                twist.linear.x -= 0.1
            elif word == 'a':
                twist.angular.z += 0.1
            elif word == 'd':
                twist.angular.z -= 0.1
            elif word == 's':
                twist.angular.z = 0
                twist.linear.x = 0
            print("Now speed:[{} cm/s],angular:[{} degree/s]".format(str(twist.linear.x), str(twist.angular.z)))
            key_pub.publish(twist)
        rate.sleep()
    # 恢复属性
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)