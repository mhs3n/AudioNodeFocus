cmake_minimum_required(VERSION 3.8)
project(audionodepackage)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
	
install(DIRECTORY launch DESTINATION share/audionodepackage/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake REQUIRED)
# find the ROS message code generators
 find_package(rosidl_default_generators REQUIRED)

# declare the message files to generate code for
 set(msg_files
 )

# declare the service files to generate code for
 #@

 rosidl_generate_interfaces(${PROJECT_NAME}
   ${msg_files}
   "srv/Texttospeech.srv"
   "srv/Record.srv"
   "srv/Stoprecording.srv"
   "srv/Detect.srv"
   "srv/Realdetect.srv"
   "srv/Stopdetect.srv"
 )
endif()

install(PROGRAMS  
  audionodepackage/service.py  
  audionodepackage/client.py
  audionodepackage/service_second_model.py
  DESTINATION lib/${PROJECT_NAME} )

ament_package()
