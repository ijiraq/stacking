import os

from setuptools import setup, find_packages

dependencies = ['requests >= 2.7',
                'astropy >= 3.0.0',
                'numpy',
                'scipy',
                'mp_ephem',
                'sip_tpv']

# Build the list of tools and scripts to be installed.
script_dirs = ['scripts']
scripts = []
for script_dir in script_dirs:
    for script in os.listdir(script_dir):
        if script[-1] in ["~", "#"]:
            continue
        scripts.append(os.path.join(script_dir, script))

console_scripts = [ 'shift_and_stack = shift_and_stack:main']

setup(name='shift_and_stack',
      version='0.1.dev1',
      author='''JJ Kavelaars (jjk@uvic.ca)''',
      maintainer='JJ Kavelaars',
      maintainer_email='jjk@uvic.ca',
      description="module of things to provide shift+stack for moving sources",
      classifiers=['Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Development Status :: 3 - Alpha',
                   'Programming Language :: Python :: 3 :: Only',
                   'Operating System :: MacOS :: MacOS X',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   ],
      scripts=scripts,
      entry_points={'console_scripts': console_scripts},
      packages=find_packages(exclude=['tests', ]),
      install_requires=dependencies
      )
