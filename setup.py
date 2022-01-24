from setuptools import setup
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(
    name='DACOpt',
    version='0.0.1',
    packages=['dacopt', 'dacopt.stac'],
    url='',
    license='GPL-3.0 License',
    author='Hidden for Anonymity',
    author_email='Hidden for Anonymity',
    description='DACOpt'
)
