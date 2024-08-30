import rclpy
from rclpy.node import Node
from audionodepackage.srv import Detect
class AudioClassificationClient(Node):
    def __init__(self):
        super().__init__('audio_class')
        self.classify_client = self.create_client(Detect, 'detect')  
        self.wait_for_service()

    def wait_for_service(self):
        while not self.classify_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for classification service to be available...')

    def send_request(self):
        user_input = input(" provide a file path: ")
        request.request = user_input
        future = self.classify_client.call_async(request)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Classification result: {response.response}")
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    client = AudioClassificationClient()
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
