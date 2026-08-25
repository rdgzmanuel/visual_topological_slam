from setuptools import find_packages, setup

package_name: str = "vts_core"

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
    description="Dataset-agnostic core algorithms for visual topological mapping.",
    license="MIT",
    python_requires=">=3.12,<3.13",
    entry_points={
        "console_scripts": [
        ],
    },
)
