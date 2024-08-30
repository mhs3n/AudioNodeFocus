#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
#
from audionodepackage.srv import Say

class StringServiceClient(Node):
    def __init__(self):
        super().__init__('string_service_client')
        self.client = self.create_client(Say, 'say')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service to be available...')
        self.send_request()

    def send_request(self):
        request = Say.Request()
        request.request,request.language  = input("Enter something to say: ")
        self.future = self.client.call_async(request.language)
        self.future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            self.get_logger().info('Response from server: %s' % response.response)
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    client = StringServiceClient()
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
