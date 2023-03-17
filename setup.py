from setuptools import setup

setup(name='geedatasets',
      install_requires=['matplotlib','numpy', 'pandas','joblib',
                        'progressbar2', 'psutil', 'scipy', 'shapely',
                        'geopandas', 'pyproj', 'rasterio'
                       ],
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      scripts=[],
      entry_points={
            "console_scripts": [
                  "geed = geedatasets.main:main",
            ],      
      },
      packages=['geedatasets'],
      include_package_data=True,
      zip_safe=False)
