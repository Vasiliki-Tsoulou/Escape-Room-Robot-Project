import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('myagv')
    pkg_description_dir = get_package_share_directory('myagv_description')
    
    urdf_file = os.path.join(pkg_description_dir, 'urdf', 'myAGV.urdf')
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'my_config.rviz')

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        )
    ])