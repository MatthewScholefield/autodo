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
        'gluoncv',
        'matplotlib',
        'lazy',
        'opencv-python',
        'prettyparse',
        'pyvips',
        'torch', 'torchvision'
    ],
    entry_points={
        'console_scripts': [
            'autodo-predict-boxes=autodo.scripts.predict_boxes:main',
            'autodo-crop=autodo.scripts.crop:main',
            'autodo-train-stage-two=autodo.scripts.train_stage_two:main',
            'autodo-run-stage-two=autodo.scripts.run_stage_two:main',
            'autodo-train-stage-three=autodo.scripts.train_stage_three:main',
            'autodo-run-stage-three=autodo.scripts.run_stage_three:main'
        ]
    }
)
