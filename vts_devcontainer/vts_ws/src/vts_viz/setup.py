from setuptools import find_packages, setup

package_name: str = "vts_viz"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Manuel Rodriguez Villegas",
    maintainer_email="manuelrodriguez@alu.comillas.edu",
    description="Optional floorplan visualization utilities.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "fit_floorplan_transform = vts_viz.fit_floorplan_transform:main",
        ],
    },
)
