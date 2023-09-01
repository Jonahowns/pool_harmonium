from setuptools import setup

setup(
    name="pool",
    version="0.0.1",
    description="Pytorch Model of Pooling Convolutional Restricted Boltzmann Machines for Sequence Analysis",
    py_modules=["rbm_torch"],
    package_dir={'': "src"}
)