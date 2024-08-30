#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from audionodepackage.srv import Texttospeech, Record, Stoprecording, Detect, Realdetect, Stopdetect
import pyaudio
import wave
import os
import librosa
import threading
import tensorflow as tf
#import tensorflow_io as tfio  # uncomment later
#import tensorflow_hub as hub
import csv
from scipy.io import wavfile
import scipy
from scipy import signal
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
import pyttsx3
from collections import defaultdict
from std_msgs.msg import String
import tensorflow_io as tfio
import pandas as pd
import csv

output_path = os.path.expanduser('src/audionodepackage/resources/recordings')
model = tf.saved_model.load("src/audionodepackage/resources/model/yamnet_ray") #put path to model here and in realtime detection bellow
print("**Server is Ready**")
loaded_model=tf.keras.models.load_model('src/resources/Transfer_learning_yamnet/saving_costum_yamnet/my_model.keras')

class StringServiceServer(Node):
    def __init__(self):
        super().__init__('string_service_server')
        self.srv_say = self.create_service(Texttospeech, 'texttospeech', self.say_callback)
        self.srv_record = self.create_service(Record, 'record', self.record_callback)
        self.srv_stop = self.create_service(Stoprecording, 'stop_recording', self.stop_callback)
        self.srv_detect_realtime = self.create_service(Realdetect, 'detect_realtime', self.detect_realtime_callback)
        self.srv_stop_realtime = self.create_service(Stopdetect, 'stop_realtime_detection', self.stop_realtime_detection_callback)
        self.srv_detect = self.create_service(Detect, 'detect', self.detect_callback)
        #creating topics for dettection
        self.silence_pub = self.create_publisher(String, 'silence_detected', 10)
        self.beep_pub = self.create_publisher(String, 'beep_detected', 10)
        self.speech_pub = self.create_publisher(String, 'speech_detected', 10)
        self.music_pub = self.create_publisher(String, 'music_detected', 10)
        self.ambient_pub = self.create_publisher(String, 'ambient_detected', 10)
        

        self.engine = pyttsx3.init()  # initialize text to speech
        
        
        #Parametr for input device

        self.declare_parameter('input_device_number', 0)
        input_device_number = self.get_parameter('input_device_number').get_parameter_value().integer_value
        self.get_logger().info(f'Using input device number: {input_device_number}')
        self.manager = RealTimeDetectionManager(self)#, input_device_number

        self.stream = None
        self.recording = False
        self.detecting = False
        self.frames = []
        self.output_filename = ""
        self.record_thread = None
        self.detect_thread = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def detect_realtime_callback(self, request, response):
        self.p = pyaudio.PyAudio()  # initialize recorder
        self.get_logger().info('Detect real-time callback triggered')
        result = self.manager.start_detection()
        response.success = result
        return response

    def stop_realtime_detection_callback(self, request, response):
        self.get_logger().info('Stop real-time detection callback triggered')
        result = self.manager.stop_detection()
        response.success = result
        return response

    def say_callback(self, request, response):
        self.get_logger().info('Incoming request to speak: %s' % request.text)
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('voice', request.language)
        self.engine.say(request.text)
        self.engine.runAndWait()
        response.success = True
        return response

    def record_callback(self, request, response):
        self.p = pyaudio.PyAudio()  # initialize recorder

        
        with self.lock:
            if self.recording:
                response.message = 'Recording is already in progress.'
                return response

            self.get_logger().info('Incoming request to record:')
            
            self.output_filename = request.filepath
            self.recording = True
            self.frames = []

            self.record_thread = threading.Thread(target=self.record_audio, args=(request.duration,))
            self.record_thread.start()

            if request.duration > 0:
                self.record_thread.join()
                response.message = f'Server finished recording and file saved to: {os.path.join(output_path, self.output_filename)}'
                response.success= True
            else:
                response.message = 'Recording without duration specified. Call stop_recording to stop.'
                response.success= True

        return response

    def record_audio(self, duration):
    
        #input_device_number = self.get_parameter('input_device_number').get_parameter_value().integer_value

        CHUNK = 4096
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        self.frames = []
        self.actual_sample_rate = 16000
        
        #for sample_rate in sample_rates:
        try:

            self.stream = self.p.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    #input_device_index=input_device_number,
                                    frames_per_buffer=CHUNK)
            
            # self.get_logger().info(f'Recording with sample rate: {sample_rate} Hz')
            # self.actual_sample_rate = sample_rate

            if duration > 0:
                for _ in range( int(RATE / CHUNK * duration)):
                    data = self.stream.read(CHUNK, exception_on_overflow=True)
                    self.frames.append(data)
                self.stop_recording()
            else:
                while self.recording:
                    data = self.stream.read(CHUNK, exception_on_overflow=True)
                    self.frames.append(data)
            
            self.get_logger().info('Recording finished.')
            return

        except Exception as e:
            self.get_logger().info(f"Error with sample rate {RATE}: {e}")
            self.p.terminate()
            self.p = pyaudio.PyAudio()

        self.get_logger().error("Could not find a working sample rate.")

    def stop_callback(self, request, response):
        with self.lock:
            if not self.recording:
                response.success = False
                return response

            self.stop_recording()
            response.success = True

        return response

    def stop_recording(self):
        self.get_logger().info('Stopping recording...')
        self.get_logger().error(f"Error with sample rate {self.actual_sample_rate}")
        self.recording = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

        output_file_path = os.path.join(output_path, self.output_filename)
        
        with wave.open(output_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.actual_sample_rate)  # Store the actual sample rate used
            wf.writeframes(b''.join(self.frames))

        self.get_logger().info(f"Audio saved to {output_file_path}")
        
    #     # Resample the audio if the actual sample rate is different from the desired rate (e.g., 44100 Hz)
    #     desired_sample_rate = 44100
    #     if self.actual_sample_rate != desired_sample_rate:
    #         self.resample_audio(output_file_path, desired_sample_rate)

    # def resample_audio(self, file_path, desired_sample_rate):
    #     self.get_logger().info(f"Resampling audio to {desired_sample_rate} Hz")
    #     try:
    #         data, sample_rate = sf.read(file_path)
    #         resampled_data = librosa.resample(data, orig_sr=sample_rate, target_sr=desired_sample_rate)
    #         sf.write(file_path, resampled_data, desired_sample_rate)
    #         self.get_logger().info(f"Resampled audio saved to {file_path}")
    #     except Exception as e:
    #         self.get_logger().error(f"Error resampling audio: {e}")


    def detect_callback(self, request, response):
        self.get_logger().info(f"Detecting on file: {request.filepath}")

        def class_names_from_csv(class_map_csv_text):
            class_names = []
            with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    class_names.append(row['display_name'])
            return class_names

        def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
            if len(waveform.shape) > 1:
                # Average the channels to convert to mono
                waveform = np.mean(waveform, axis=1)
            
            if original_sample_rate != desired_sample_rate:
                desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
                waveform = scipy.signal.resample(waveform, desired_length)
            return desired_sample_rate, waveform

        class_map_path = model.class_map_path().numpy()
        class_names = class_names_from_csv(class_map_path)
        # Load the audio file and get the duration

        
        wav_file_name = request.filepath
        sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
        waveform = wav_data / tf.int16.max
        track_duration = librosa.get_duration(y=waveform, sr=sample_rate)
        scores, embeddings, spectrogram = model(waveform)
        result = loaded_model(embeddings).numpy()
        my_classes = ['Speech', 'Silence', 'Music', 'Beep', 'Robot_moving']
        inferred_class = my_classes[result.mean(axis=0).argmax()]
        top_class_threshold = 0.4
        window_duration = 1  # Each window is 0.96 seconds
        presence_intervals = defaultdict(list)
        top_scores_debug = []
        intervals = []

        try:


# Get the model predictions

            for i in range(round(track_duration)//window_duration+window_duration+1):
                window_start = i-window_duration-1 * window_duration
                window_end = min(window_start + window_duration, track_duration)  # Ensure window_end does not exceed track duration
                top_class = np.argmax(scores[i])
                top_score = scores[i][top_class]
                
                # Debug: Store top scores for each window
                top_scores_debug.append((top_class, top_score))
                
                if top_score >= top_class_threshold:
                    class_name = inferred_class 
                    presence_intervals[class_name].append((max(window_start,0), max(0,window_end)))

            # Merge contiguous intervals for the same class
            merged_intervals = defaultdict(list)
            for inferred_class, intervals in presence_intervals.items():
                merged = []
                current_interval = intervals[0]
                
                for interval in intervals[1:]:
                    if interval[0] <= current_interval[1]:
                        current_interval = (current_interval[0], max(current_interval[1], interval[1]))
                    else:
                        merged.append(current_interval)
                        current_interval = interval
                merged.append(current_interval)
                merged_intervals[class_name] = merged

            # Calculate the total duration of presence for each class
            total_durations = {class_name: sum(end - start for start, end in intervals)
                            for class_name, intervals in merged_intervals.items()}

            # Prepare the final result
            result = []
            for class_name, intervals in merged_intervals.items():
                interval_str = "U".join([f"[{start:.2f}s, {end:.2f}s]" for start, end in intervals])
                result.append((class_name, interval_str))

            # Sort by total presence duration
            sorted_result = sorted(result, key=lambda x: total_durations[x[0]], reverse=True)

            # Display the result
            for inferred_class, intervals in sorted_result:
                print(f"Class: {inferred_class}, Intervals: {intervals}")
                        # Return the formatted response
            response_str = "\n".join([f"Class: {inferred_class}, Intervals: {intervals}" for inferred_class, intervals in sorted_result])
            response.detection = response_str
            return response

        except Exception as e:
            self.get_logger().error(f"Error processing scores: {e}")
            response.detection = f"Error: {e}"
            return response





class RealTimeDetectionManager:
    def __init__(self, node):#,input_device_number
        self.node = node
        self.p = pyaudio.PyAudio()
        #self.stream = None
        self.detecting = False
        self.detect_thread = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        #self.input_device_number = input_device_number
        self.yamnet_model = tf.saved_model.load("src/audionodepackage/resources/model/yamnet_ray")
        self.loaded_model=tf.keras.models.load_model('src/resources/Transfer_learning_yamnet/saving_costum_yamnet/my_model.keras')                                                                 #path to model
        self.class_names = self._class_names_from_csv(model.class_map_path().numpy())
        @tf.function
        def load_wav_16k_mono(filename):
            """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
            file_contents = tf.io.read_file(filename)
            wav, sample_rate = tf.audio.decode_wav(
                file_contents,
                desired_channels=1)
            wav = tf.squeeze(wav, axis=-1)
            sample_rate = tf.cast(sample_rate, dtype=tf.int64)
            wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
            
            return wav
        self.my_classes = ['Speech', 'Silence', 'Music', 'Beep', 'Robot_moving']
        class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names =list(pd.read_csv(class_map_path)['display_name'])
        


    def start_detection(self):
        with self.lock:
            if self.detecting:
                return 'Real-time detection is already in progress.'

            self.node.get_logger().info('Starting real-time detection...')
            self.detecting = True
            self.stop_event.clear()

            self.detect_thread = threading.Thread(target=self._detect_audio)
            self.detect_thread.start()
            return True

    def stop_detection(self):
        with self.lock:
            if not self.detecting:
                return 'No real-time detection is in progress.'

            self.node.get_logger().info('Stopping real-time detection...')
            self.detecting = False
            self.stop_event.set()

            # Wait for the thread to exit
            if self.detect_thread is not None:
                self.detect_thread.join()
                self.detect_thread = None

            self._stop_stream()
            return True

    def _detect_audio(self):
        chunk = 4096
        format = pyaudio.paInt16
        channels = 1
        # sample_rates = [16000, 44100, 48000]  # Add more sample rates if needed

        #for sample_rate in sample_rates:
        try:
            self.stream = self.p.open(format=format,
                                    channels=channels,
                                    rate=16000,
                                    input=True,
                                    #input_device_index=self.input_device_number,  # Ensure this is set correctly
                                    frames_per_buffer=chunk,
                                    stream_callback=self._audio_callback)
            self.sample_rate = 16000  # Store the sample rate
            self.node.get_logger().info(f"Successfully opened stream with sample rate: {16000}")
            
        except OSError as e:
            self.node.get_logger().error(f"Failed to open stream with sample rate {16000}: {e}")
        # else:
        #     self.node.get_logger().error("Failed to open audio stream with all tested sample rates.")
        #     return

        self.stream.start_stream()

        # Loop until detection is stopped
        while self.detecting and not self.stop_event.is_set():
            # Allow other ROS2 nodes to process their callbacks
            rclpy.spin(self.node, timeout_sec=0.5)

        self._stop_stream()
        self.node.get_logger().info('Real-time detection thread stopped.')


    def _stop_stream(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def _audio_callback(self, in_data, frame_count, time_info, status):
            # Convert the audio data to a NumPy array
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
            
            # Resample audio data if needed
            target_sample_rate = 16000
            current_sample_rate = self.sample_rate  # Get the sample rate of the stream
            if current_sample_rate != target_sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=current_sample_rate, target_sr=target_sample_rate)
            
            # Ensure audio_data has the correct shape
            audio_data = np.reshape(audio_data, (len(audio_data),))
            
            # Model inference
            scores, embeddings, spectrogram = self.yamnet_model(audio_data)
            scores_np = scores.numpy()
            result = self.loaded_model(embeddings).numpy()
            inferred_class = self.my_classes[result.mean(axis=0).argmax()]

            # Get the top predicted class
            mean_scores = np.mean(scores_np, axis=0)
            top_class_idx = mean_scores.argmax()
            top_class = self.class_names[top_class_idx]
            top_score = mean_scores[top_class_idx]
            if inferred_class == 'Silence':
                self.node.silence_pub.publish(String(data='Silence detected'))
            elif inferred_class == 'Beep':
                self.node.beep_pub.publish(String(data='Beep detected'))
            elif inferred_class == 'Speech':
                self.node.speech_pub.publish(String(data='Speech detected'))
            elif inferred_class == 'Music':
                self.node.music_pub.publish(String(data='Music detected'))
            elif inferred_class == 'Robot_moving':
                self.node.ambient_pub.publish(String(data='Robot_moving'))

            # Define the threshold
            score_threshold = 0.3  # Adjust this value as needed

            # Print the top class if the score is above the threshold
            #if top_score >= score_threshold:
            self.node.get_logger().info(f"The main sound is: {inferred_class} ")

            return (in_data, pyaudio.paContinue)
    def _class_names_from_csv(self, class_map_csv_text):
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names




def main(args=None):
    rclpy.init(args=args)
    string_service_server = StringServiceServer()
    rclpy.spin(string_service_server)
    string_service_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
