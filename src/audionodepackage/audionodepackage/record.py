#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from audionodepackage.srv import Record, Stoprecording

class StringServiceClient(Node):
    def __init__(self):
        super().__init__('string_service_client')
        self.client_record = self.create_client(Record, 'record')
        self.client_stop = self.create_client(Stoprecording, 'stop_recording')
        while not self.client_record.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for record service to be available...')
        while not self.client_stop.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for stop recording service to be available...')
        self.send_request()

    def send_request(self):
        request = Record.Request()
        inputs = input("Enter file name and optionally duration (e.g., 'filename.wav 10' or just 'filename.wav'): ").split()
        request.request = inputs[0]
        request.duration = float(inputs[1]) if len(inputs) > 1 else 0.0
        self.future = self.client_record.call_async(request)
        self.future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            self.get_logger().info('Response from server: %s' % response.response)
            if response.response.startswith("Recording without duration"):
                input("Press Enter to stop recording...")
                stop_request = Stoprecording.Request()
                stop_future = self.client_stop.call_async(stop_request)
                stop_future.add_done_callback(self.handle_stop_response)
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

    def handle_stop_response(self, future):
        try:
            response = future.result()
            self.get_logger().info('Stop recording response from server: %s' % response.success)
        except Exception as e:
            self.get_logger().error('Stop recording service call failed %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    client = StringServiceClient()
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
