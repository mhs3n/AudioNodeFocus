from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'audionodepackage'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mhsn',
    maintainer_email='soulirayen@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'server_launch = audionodepackage.server_launch:main',
            'server = audionodepackage.service:main',
            'speak = audionodepackage.client:main',
            'record = audionodepackage.record:main',
            'stop_record = audionodepackage.record:main',
            'detect = audionodepackage.detect:main',
            'beep = audionodepackage.Beep_listner:main',
            'silence = audionodepackage.Silence_listner:main',
            'speech= audionodepackage.Speech_listner:main',
            'ambiant= audionodepackage.Ambiant_listner:main',
            'music= audionodepackage.Music_listner:main'
        ]
    },
)
