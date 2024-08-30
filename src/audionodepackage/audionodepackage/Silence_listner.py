#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ListenerNode(Node):
    def __init__(self):
        super().__init__("listener_silence")
        self.get_logger().info("Listener Python node has been created")
        self.subscriber_ = self.create_subscription(String, 'silence_detected', self.subscriber_callback, 10)

    def subscriber_callback(self, msg):
        data = msg.data
        self.get_logger().info(f'received data: {data}')

def main(args=None):
    rclpy.init(args=args)
    node = ListenerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__=='__main__':
    main()