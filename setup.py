import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="signal_backtester",
    version="0.0.1",
    author="Alex Chao",
    author_email="alexchao86@gmail.com",
    description="A small example package",
    long_description="backtest signals and perform signal univariate studies",
    long_description_content_type="text/markdown",
    url="https://github.com/alexhchao/signal_backtester",
    test_suite='nose.collector',
    tests_require=['nose'],
    packages=setuptools.find_packages(),
)

