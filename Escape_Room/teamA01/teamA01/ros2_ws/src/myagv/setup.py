import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'myagv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ROS 2 Resource & Launch installations
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fotisg',
    maintainer_email='fotisg@todo.todo',
    description='MyAGV navigation, control, and mapping pipeline package.',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'a_star_node = myagv.a_star_node:main',
            'lidar_mapper_node = myagv.lidar_mapper_node:main',
            'vision_node = myagv.vision_node:main',
            'watchdog = myagv.watchdog:main',
            'holonomic_velocity_controller = myagv.holonomic_velocity_controller:main',
            'task_manager_node = myagv.task_manager_node:main',
            'v4l2_camera_node = myagv.v4l2_camera_node:main',
        ],
    },
)
