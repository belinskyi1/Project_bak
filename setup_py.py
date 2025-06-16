from setuptools import setup, find_packages

setup(
    name="xray-object-classification",
    version="1.0.0",
    author="Yaroslav Belinsky",
    description="X-ray object classification using ResNet-50",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "PyYAML>=6.0.0",
        "tqdm>=4.66.0",
    ],
)