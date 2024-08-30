from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import subprocess

def get_device_id():
    result = subprocess.run(['python3', 'script/get_last_id.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return '0'  # Default or error ID if needed

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('input_device_number', default_value=get_device_id(), description='Input device number'),
        
        Node(
            package='audionodepackage',
            executable='service_second_model.py',
            name='StringServiceServer',
            output='screen',
            parameters=[{
                'input_device_number': LaunchConfiguration('input_device_number')
            }]
        ),
    ])
