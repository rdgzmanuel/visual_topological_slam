from setuptools import find_packages, setup

package_name: str = "vts_mapping"

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
    description="Topological graph building node.",
    license="MIT",
    python_requires=">=3.12,<3.13",
    entry_points={
        "console_scripts": [
            "graph_builder = vts_mapping.graph_builder_node:main",
        ],
    },
)
