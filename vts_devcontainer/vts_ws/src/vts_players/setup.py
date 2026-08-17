from setuptools import find_packages, setup

package_name: str = "vts_players"

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
    description="Dataset adapters for COLD and CID-SIMS.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "cold_player = vts_players.cold_player_node:main",
            "cid_sims_player = vts_players.cid_sims_player_node:main",
        ],
    },
)
