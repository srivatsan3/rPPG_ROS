from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rppg_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_dir={'':'.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'), glob('launch/*.launch.py')),
        (os.path.join('share',package_name,'config'), glob('config/*.yaml')),
        (os.path.join('share',package_name,'msg'), glob('msg/*.msg')),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mscrobotics2425laptop11',
    maintainer_email='srivatsanraman3@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webcam_reader = rppg_node.webcam_reader:main',
            'rppg_toolbox_node_run = rppg_node.rppg_toolbox_node_run:main' ,
            'rppg_toolbox_node_run_nn = rppg_node.rppg_toolbox_node_run_nn:main' ,
            'rppg_toolbox_video_node = rppg_node.rppg_toolbox_video_node:main'
        ],
    },
)
