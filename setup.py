import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepsleep",
    version="1.0.3",
    author="Ami Beuret",
    author_email="amibeuret@gmail.com",
    description="Sleep EEG classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://sleeplearning.ethz.ch",
    packages=setuptools.find_packages(),
    scripts=['predict.py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch==1.6.0",
        "numpy==1.18.1",
        "pandas==0.25.3",
        "Pillow==6.2.2",
        "scipy==1.4.1",
        "sklearn==0.0",
        "torchvision==0.7.0",
        "matplotlib==3.1.2",
        "mne==0.19.2",
        "pyEDFlib==0.1.22",
        "PyYAML==5.3",
        "h5py==2.10.0",
        "mlflow==1.5.0",
        "hmmlearn==0.2.3"
    ],
    python_requires='>=3.6',
)
