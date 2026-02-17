from setuptools import find_packages, setup

package_name = 'vts_commands'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Manuel Rodriguez',
    maintainer_email='manuel.rodriguezvillegas09@gmail.com',
    description='Natural language commands node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "commands = vts_commands.commands_node:main"
        ],
    },
)
