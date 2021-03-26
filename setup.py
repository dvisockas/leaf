import os
import setuptools

VERSION = '0.0.1'

folder = os.path.dirname(__file__)
with open(os.path.join(folder, 'requirements.txt')) as fp:
  install_requires = [line.strip() for line in fp]


description = ('PyTorch implementation of LEAF')

setuptools.setup(
    name='leaf_torch',
    version=VERSION,
    packages=setuptools.find_packages(),
    description=description,
    long_description=description,
    url='https://github.com/dvisockas/leaf',
    author='Danielius Visockas',
    author_email='danieliusvisockas@gmail.com',
    install_requires=install_requires,
    license='MIT',
    keywords='leaf learnable audio frontend gabor convolution pytorch',
)
