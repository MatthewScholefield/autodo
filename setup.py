from setuptools import setup

setup(
    name='autodo',
    version='0.1.0',
    description='Car position predictor tool',
    url='https://github.com/matthewscholefield/autodo',
    author='Matthew D. Scholefield',
    author_email='matthew331199@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='autodo',
    packages=['autodo'],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'mxnet',
        'gluoncv',
        'matplotlib',
        'lazy',
        'opencv-python'
    ],
    entry_points={
        'console_scripts': [
            'autodo=autodo.__main__:main'
        ],
    }
)
