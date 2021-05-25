## Contributing

Setup the development environment by running the following on your terminal:

```bash
git clone https://github.com/diogoseca/timegym.git # clone the repository
cd timegym/ # go to the project root
pip install flit # install flit if you don't have it already
flit install # install the package for development
pytest --nbval-lax examples/*.ipynb  # test the notebooks with coverage
```

If you wish to test the coverage of the notebooks tested, then run:

```bash
coverage run -m pytest --nbval-lax examples/*.ipynb
```

In case of doubt follow the awesome guidelines of https://requests.kennethreitz.org/