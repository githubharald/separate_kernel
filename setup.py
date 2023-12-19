from setuptools import setup

setup(
    name='separate-kernel',
    version='1.0.0',
    description='Separate 2D convolution kernels.',
    author='Harald Scheidl',
    packages=['separate_kernel'],
    install_requires=['scipy', 'numpy'],
    python_requires='>=3.9'
)
