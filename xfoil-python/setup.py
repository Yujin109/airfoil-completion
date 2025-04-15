from setuptools import setup, find_packages

version = '0.0.16'

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='xfoil',
    version=version,
    description='Stripped down version of XFOIL as compiled python module ',
    #long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Fortran',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
    ],
    keywords='xfoil airfoil aerodynamic analysis',
    #url='https://github.com/daniel-de-vries/xfoil-python',
    #download_url='https://github.com/daniel-de-vries/xfoil-python/tarball/' + version,
    author='Daniël de Vries',
    author_email='contact@daniel-de-vries.com',
    license='GNU General Public License v3 or later (GPLv3+)',
    packages=['xfoil'],
    package_dir={'': 'src'},
    #ext_modules=[CMakeExtension('xfoil.xfoil')],
    #cmdclass={'build_ext': CMakeBuild},
    install_requires=['numpy'],
    #zip_save=False
)
