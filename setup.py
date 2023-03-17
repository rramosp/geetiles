from setuptools import setup

setup(name='geetiles',
      install_requires=['matplotlib','numpy', 'pandas','joblib',
                        'progressbar2', 'psutil', 'scipy', 'shapely',
                        'geopandas', 'pyproj', 'rasterio'
                       ],
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      scripts=[],
      entry_points={
            "console_scripts": [
                  "geet = geetiles.main:main",
            ],      
      },
      packages=['geetiles'],
      include_package_data=True,
      zip_safe=False)
