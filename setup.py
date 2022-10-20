from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='prml-machinelearning',
    version='v0.3.0',
    author='Matthew Johnson, Leigh Hill',
    author_email='greenchicken1902@gmail.com',
    maintainer='Matthew Johnson',
    maintainer_email='greenchicken1902@gmail.com',
    description='Coded responses for each assignment question',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GreenMachine582/PRML-MachineLearning',
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only'
    ],
    keywords='prml, machine-learning, cross-validation, regression, classification',
    python_requires='>=3.10, <4',
)
