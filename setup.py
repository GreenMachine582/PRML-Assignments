from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='prml-MachineLearning',
    version='v0.2.3',
    author='Matthew Johnson, Leigh Hill',
    author_email='greenchicken1902@gmail.com, u3215513@uni.canberra.edu.au',
    maintainer='Matthew Johnson',
    maintainer_email='greenchicken1902@gmail.com',
    description='Coded responses for each assignment question',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GreenMachine582/PRML-MachineLearning',
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only'
    ],
    keywords='prml, machine-learning, cross-validation, , classifiers, estimators',
    python_requires='>=3.10, <4',
)
