from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='mgn',
      version='0.0.1',
      url='http://github.com/robinhenry/meng',
      author='Robin Henry',
      description="My Master Thesis (MEng) code.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author_email='robin@robinxhenry.com',
      packages=['meng'],
      install_requires=[],
      python_requires='>=3.6'
      )
