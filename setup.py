from setuptools import setup
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(
    name='DACOpt',
    version='0.0.1',
    packages=['dacopt', 'dacopt.stac'],
    url='',
    license='GPL-3.0 License',
    author='Duc Anh Nguyen',
    author_email='d.a.nguyen@liacs.leidenuniv.nl',
    description='DACOpt'
)
