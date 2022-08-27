from setuptools import setup, find_packages
import subprocess
import os


def get_version():
    latest_version = (
        subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )

    if "-" in latest_version:
        x = latest_version.split("-")
        v, i, s = x[0], x[-2], x[-1]
        if len(x) == 2:
            i = 0
        return f"{v}+{i}.git.{s}"
    return latest_version


version = get_version()

assert "-" not in version
assert "." in version

with open("src/VERSION", "w", encoding="utf-8") as fh:
    fh.write("%s\n" % version)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='PRML-MachineLearning',
    version=version,
    author='Matthew Johnson, Leigh Hill',
    author_email='greenchicken1902@gmail.com, LeighHillx@gmail.com',
    maintainer='Matthew Johnson',
    maintainer_email='greenchicken1902@gmail.com',
    description='Coded responses for each assignment question',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GreenMachine582/PRML-MachineLearning',
    packages=find_packages(),
    package_data={'src': ['VERSION']},
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
    keywords='PRML, machine-learning, cross-validation, long-short-term-memory, convolutional-neural-networks, linear-regression, k-means-clustering',
    python_requires='>=3.10, <4',
    entry_points={},
    install_requires=[],
)
