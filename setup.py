from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()


with open("requirements.txt") as f:
    requirements = f.readlines()


setup(
    name='causalml',
    author='Huigang Chen, Totte Harinen, Jeong-Yoon Lee, Mike Yung, Zhenyu Zhao',
    author_email='',
    description='Python Package for Uplift Modeling and Causal Machine Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=2.7',
    url='https://github.com/uber-common/causalml/',
    version='0.1.0',
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=requirements
)
