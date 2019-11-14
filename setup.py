from setuptools import setup

setup(name='pattern_finder_gpu',
      version='1.0',
      description='Brute force OpenCL based pattern localization in images that supports masking and weighting.',
      url='https://github.com/HearSys/pattern_finder_gpu',
      author='Samuel John (HörSys GmbH)',
      author_email='john.samuel@hoersys.de',
      license='MIT',
      packages=['pattern_finder_gpu'],
      install_requires=['pyopencl', 'numpy', 'scipy', 'matplotlib', 'skimage'],
      zip_safe=False)
