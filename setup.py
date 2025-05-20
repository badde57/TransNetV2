from setuptools import setup

setup(
    name="transnetv2",
    version="1.0.0",
    # let user install tensorflow, etc. manually
    # install_requires=[
    #     "tensorflow>=2.0",
    #     "ffmpeg-python",
    #     "pillow"
    # ],
    entry_points={
        "console_scripts": [
            # "transnetv2_predict = transnetv2.transnetv2:main", # Path no longer valid with inference-pytorch
        ]
    },
    packages=["transnetv2"],
    package_dir={"transnetv2": "inference-pytorch"},
    package_data={"transnetv2": ["transnetv2-pytorch-weights.pth"]},
    zip_safe=False
)
