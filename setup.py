from setuptools import setup, find_packages

setup(name="ait",
      version="0.0.1",
      author="Tao Jing",
      author_email="jingt20@mails.tsinghua.edu.cn",
      description="astro image toolkits",
      packages=find_packages(),
      install_requires=[
          'numpy', 'joblib', 'astropy', 'pandas', 'scikit-image', 'reproject',
          'h5py'
      ])
