from distutils.core import setup

setup(
    name='yosegi',
    version='0.0.0',
    description='Standard dataset operations',
    author='Ryo Miyajima',
    url='https://github.com/sergeant-wizard/yosegi',
    packages=[
        'yosegi',
        'yosegi.io',
    ],
    package_data={
        'yosegi': ['py.typed'],
    },
)
