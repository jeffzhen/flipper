from distutils.core import setup, Extension
import os, glob, numpy



__version__ = '0.0.1'

def indir(dir, files): return [dir+f for f in files]
def globdir(dir, files):
    rv = []
    for f in files: rv += glob.glob(dir+f)
    return rv

setup(name = 'flipper',
    version = __version__,
    description = __doc__,
    long_description = __doc__,
    license = 'GPL',
    author = 'Jeff Zheng',
    author_email = '',
    url = '',
    package_dir = {'flipper':'src'},
    packages = ['flipper'],

    scripts = glob.glob('scripts/*'),
)

