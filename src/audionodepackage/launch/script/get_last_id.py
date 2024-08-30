import subprocess
import re
import pyaudio

def get_last_arecord_index():
    # Step 1: Run `arecord` and capture the output
    arecord_output = subprocess.check_output("arecord -l", shell=True).decode()

    # Step 2: Extract the last card and device numbers
    matches = re.findall(r'card (\d+):.*device (\d+):.*\[([^\]]+)\]', arecord_output)
    if matches:
        last_card, last_device, device_name = matches[-1]  # Get the last match
    else:
        return None  # No matching devices found

    # Step 3: Find the corresponding PyAudio device index
    p = pyaudio.PyAudio()
    device_index = None

    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if f"hw:{last_card},{last_device}" in device_info['name']:
            device_index = i
            break

    return int(device_index)  # Return the index or None if not found

# Call the function and print the result
index = get_last_arecord_index()
if index is not None:
    print(index)  # Print just the index
else:
    print("Could not find a matching PyAudio device.")
