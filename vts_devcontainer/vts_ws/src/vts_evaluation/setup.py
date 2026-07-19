from setuptools import find_packages, setup

package_name: str = "vts_evaluation"

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
    description="Offline evaluation metrics CLI.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "evaluate_run = vts_evaluation.evaluate_run:main",
            "calibrate_floorplan = vts_evaluation.calibrate_floorplan:main",
            "plot_odometry = vts_evaluation.plot_odometry:main",
            "compare_odometry_maps = vts_evaluation.compare_odometry_maps:main",
            "place_recognition_eval = vts_evaluation.place_recognition_eval:main",
            "plot_lambda2 = vts_evaluation.plot_lambda2:main",
        ],
    },
)
