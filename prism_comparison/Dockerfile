# Thin layer over the authors' prebuilt image: only adds Python packages that
# their OpenPlaceRecognition install expects but the image lacks.
FROM kirillmouraviev/prism-topomap:cuda12.1-ros-noetic
RUN pip install --no-cache-dir loguru numpy-quaternion seaborn albumentations
