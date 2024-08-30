# Audio Node

**Audio Node** is a ROS2 audio node that provides functionalities for text-to-speech recording and audio classification. This project allows you to perform text-to-speech in multiple languages, record audio, and classify recorded audio or real-time audio.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   sudo apt-get install portaudio19-dev
   ```

3. Build the packages:

   ```bash
   colcon build
   source install/setup.bash
   ```

## Usage

1. **Start the server**: Open one terminal and run the server:

   ```bash
   ros2 run audionodepackage service.py
   ```

2. **Execute services**: Open a separate terminal to execute the services:

   ### Text-to-Speech in French:

   ```bash
   ros2 service call /texttospeech audionodepackage/srv/Texttospeech "{text: \"commençant test numero 1\", language: "french"}"

   ```

   ### Text-to-Speech in English:

   ```bash
   ros2 service call /texttospeech audionodepackage/srv/Texttospeech "{text: \"Starting test number 1\", language: \"english\"}"
   ```

   ### Record with Duration:

   ```bash
   ros2 service call /record audionodepackage/srv/Record "{filepath: \"test2\", duration: 5}"
   ```

   ### Record without Duration:

   ```bash
   ros2 service call /record audionodepackage/srv/Record "{filepath: \"test2\"}"
   ```

   ### Stop Recording:

   ```bash
   ros2 service call /stop_recording audionodepackage/srv/Stoprecording
   ```

   ### Detect on a Recorded Audio File:

   ```bash
   ros2 service call /detect audionodepackage/srv/Detect "{filepath: "src/audionodepackage/resources/recordings/Beep.wav"}"
   ```

   ### Real-Time Detection Launch:

   ```bash
   ros2 service call /detect_realtime audionodepackage/srv/Realdetect
   ```

   ### Stopping Real-Time Detection:

   ```bash
   ros2 service call /stop_realtime_detection audionodepackage/srv/Stopdetect
   ```
   ## Using Launch file:
   To use the last input device
     ```bash
   ros2 launch audionodepackage server.launch.py 
   ```
   If you want to specify which input device you want ti use you can either run [Get_last_id](src/audionodepackage/script/get_last_id.py) script or run 
   ```bash
   arecord -l
   ```
   And match the device you want to the ID provided by Pyaudio.

   This id can now be used as input with the launch file 
   ```bash
   ros2 launch audionodepackage server.launch.py input_device_number:=4 
   ```

**Note:** Once the input device is manually specified, it cannot be changed dynamically during operation. To switch to a different input device, you must restart the server.

**Note 2:** There are two service servers, each running a different model. One model is specifically trained to detect the sound of the robot moving. Due to limited data, this model may perform worse in real-time scenarios compared to the primary model. To improve its performance, refer to [this notebook](src/resources/Transfer_learning_yamnet) for detailed steps on further training.

To launch the server with the second model, use the following command:
```bash
ros2 launch audionodepackage server_second_model.launch.py
```



## Troubleshooting

For any issues or questions, please contact me via:

- **Email**: soulirayen@gmail.com
- **LinkedIn**: [Rayen Souli](https://www.linkedin.com/in/rayen-souli/)

## License

Specify the license under which your project is released here.
