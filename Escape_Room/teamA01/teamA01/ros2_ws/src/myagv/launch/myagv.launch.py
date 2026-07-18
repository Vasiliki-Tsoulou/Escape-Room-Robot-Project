import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('myagv_description')
    
    # 1. Prepare Robot Description (URDF)
    urdf_file = os.path.join(pkg_dir, 'urdf', 'myAGV.urdf')
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # 3. Base Infrastructure Nodes
    nodes_to_launch = [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
        ),

         Node(
             package='v4l2_camera',
             executable='v4l2_camera_node',
             name='v4l2_camera_node',
             output='screen',
             parameters=[{'use_sim_time': True}]
         ),
     ]


    nodes_to_launch = []
    # 4. Custom AGV Nodes List (Removes repetitive code blocks)
    custom_nodes = [
        'task_manager_node',
        'lidar_mapper_node',
        'a_star_node',
        'vision_node',
        'watchdog',
        'holonomic_velocity_controller',
    ]

    # Dynamically append custom nodes to the execution list
    for node_name in custom_nodes:
        nodes_to_launch.append(
            Node(
                package='myagv',
                executable=node_name,
                name=node_name,
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        )

    return LaunchDescription(nodes_to_launch)
