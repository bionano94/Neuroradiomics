import os

try:
    from setuptools import setup
    from setuptools import find_packages

except ImportError:
  from distutils.core import setup
  from distutils.core import find_packages
  

def get_requires (requirements_filename):
  '''
  What packages are required for this module to be executed?
  Parameters
  ----------
    requirements_filename : str
      filename of requirements (e.g requirements.txt)
  Returns
  -------
    requirements : list
      list of required packages
  '''
  with open(requirements_filename, 'r') as fp:
    requirements = fp.read()

  return list(filter(lambda x: x != '', requirements.split()))



def read_description (readme_filename):
  '''
  Description package from filename
  Parameters
  ----------
    readme_filename : str
      filename with readme information (e.g README.md)
  Returns
  -------
    description : str
      str with description
  '''

  try:

    with open(readme_filename, 'r') as fp:
      description = '\n'
      description += fp.read()

    return description

  except IOError:
    return ''

here = os.path.abspath(os.path.dirname(__file__))

#Package-Metadata
NAME = "Neuroradiomics"
DESCRIPTION = 'Package for Brain extraction and segmentation from a t1 MR Image'
URL = 'https://github.com/bionano94/Neuroradiomics'
EMAIL = 'nicolas.biondini@studio.unibo.it'
AUTHOR = 'Nicolas Biondini, Riccardo Biondi'
VERSION = None
KEYWORDS = 'radiomics brain brain-extraction skull-stripping segmentation'
REQUIREMENTS_FILENAME = os.path.join(here, 'requirements.txt')
VERSION_FILENAME = os.path.join(here, 'Neuroradiomics', '__version__.py')
README_FILENAME = os.path.join(here, 'README.md')

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
  LONG_DESCRIPTION = read_description(README_FILENAME)

except IOError:
  LONG_DESCRIPTION = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
  with open(VERSION_FILENAME) as fp:
    exec(fp.read(), about)

else:
  about['__version__'] = VERSION

# parse version variables and add them to command line as definitions
Version = about['__version__'].split('.')


setup(
    name=NAME,
    version=about['__version__'],
    author=AUTHOR,
    author_email=EMAIL,
    mantainer=AUTHOR,
    mantainer_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=URL,
    keywords=KEYWORDS,
    packages=find_packages(include=['Neuroradiomics','Neuroradiomics.*'], exclude=('Test', 'Tests')),
    include_package_data=True, # no absolute paths are allowed
    platforms='any',
    install_requires=get_requires(REQUIREMENTS_FILENAME),

    entry_points={
        'console_scripts': [
        'atlas_registration = run_registration:main',
        'skull_stripping = run_stripping:main',
    ]},

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    license = 'Apache'
)
