1. Power on the robot - the files need to be preloaded in a folder under
~/Desktop/Projects/teamA01
The Skimage python library is also needed and should be downloaded
2. The domain id needs to be on 10
- nano ~/.bashrc at the end of the file make sure it writes the correct domain
3. Setting up ROS (first terminal after connecting to er user)
- source /opt/ros/noetic/setup.bash
- roscore
4. Running odometry node on ROS (second terminal after connecting to er user)
- source /opt/ros/noetic/setup.bash
- cd Desktop/Projects/myagv_ros/
- source devel/setup.bash
- roslaunch myagv_odometry myagv_active.launch
5. Running Lidar node on ROS(third terminal after connecting to er user)
- source /opt/ros/noetic/setup.bash
- cd Desktop/Projects/myagv_ros/src/myagv_odometry/scripts
- ./start_ydlidar.sh
- cd ~/Desktop/Projects/teamA01/myagv_ros
- source devel/setup.bash
- roslaunch ydlidar_ros_driver X2.launch
6. Running the bridge(fourth terminal after connecting to er user)
- source /opt/ros/noetic/setup.bash
- cd Desktop/Projects/teamA01/
- rosparam load bridge.yaml
- source /opt/ros/galactic/setup.bash
- ros2 run ros1_bridge parameter_bridge
- Running the launch file for rviz(connected to your own user)
- source /opt/ros/galactic/setup.bash
- cd /ros2_ws(user path)
- source install/setup.bash
- ros2 launch myagv_rviz robot_bringup.launch.py
- might need this export for the urdf to show: export LIBGL_ALWAYS_SOFTWARE=1
7. Running the lidar_mapper_node(fifth terminal after connecting to er user)
- source /opt/ros/galactic/setup.bash
- cd Desktop/Projects/teamA01/ros2_ws
- colcon build –symlink-install
- source install/setup.bash
- ros2 run myagv lidar_mapper_node
8. Running the a_star_node(sixth terminal after connecting to er user)
- source /opt/ros/galactic/setup.bash
- cd Desktop/Projects/teamA01/ros2_ws
- colcon build –symlink-install
- source install/setup.bash
- ros2 run myagv a_star_node
9. Running the holonomic_velocity_controller(seventh terminal after connecting to er user)
- source /opt/ros/galactic/setup.bash
- cd Desktop/Projects/teamA01/ros2_ws
- colcon build –symlink-install
- source install/setup.bash
- ros2 run myagv holonomic_velocity_controller
10.Running the task_manager_node(eighth terminal after connecting to er user)
- source /opt/ros/galactic/setup.bash
- cd Desktop/Projects/teamA01/ros2_ws
- colcon build –symlink-install
- source install/setup.bash
- ros2 run myagv task_manager_node
