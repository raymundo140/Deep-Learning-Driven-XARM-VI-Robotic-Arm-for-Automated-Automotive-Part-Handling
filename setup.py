from setuptools import find_packages, setup

package_name = 'xarm_final'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # so that `get_package_share_directory('xarm_final')` gets registered
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # install both package.xml and your model into share/xarm_final/
        ('share/' + package_name, [
            'package.xml',
            'xarm_final/my_model.h5',   # <— add your .h5 here
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='goma',
    maintainer_email='a01285451@tec.mx',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main = xarm_final.main:main',
            'hand_tracking = xarm_final.hand_tracking:main',
            'camara = xarm_final.camara:main',
            'ss = xarm_final.ss:main',
            'break = xarm_final.break:main',
            'webcam_inference = xarm_final.webcam_inference:main',
            'pointnet_recognizer = xarm_final.pointnet_recognizer:main',
            'black_follow = xarm_final.black_follow:main',
            'black_follow_nok = xarm_final.black_follow_nok:main',
            'scanner_node = xarm_final.scanner_node:main',
            'model = xarm_final.model:main',
        ],
    },
)
