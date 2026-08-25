from setuptools import find_packages, setup

package_name: str = "vts_bringup"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/pipeline.launch.py"]),
        ("share/" + package_name + "/config", [
            "config/cold_freiburg_a.yaml",
            "config/cold_freiburg_ext.yaml",
            "config/cold_saarbruecken_a.yaml",
            "config/cold_saarbruecken_ext.yaml",
            "config/cid_sims_apartment1_1.yaml",
            "config/cid_sims_apartment2_1.yaml",
            "config/cid_sims_apartment3_1.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Manuel Rodriguez Villegas",
    maintainer_email="manuelrodriguez@alu.comillas.edu",
    description="Launch files and per-environment configurations.",
    license="MIT",
    python_requires=">=3.12,<3.13",
    entry_points={
        "console_scripts": [
        ],
    },
)
