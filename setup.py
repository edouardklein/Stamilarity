from distutils.core import setup
setup(
    name = "stamilarity",
    packages = ["stamilarity"],
    version = "0.0.1",
    description = "Quantify the statistical similarity of experimental samples",
    author = "Edouard Klein",
    author_email = "edou -at- rdklein.fr",
    url = "https://stamilarity.readthedocs.org",
    download_url = "https://github.com/edouardklein/stamilarity",
    keywords = ["statistics"],
    classifiers = [
        "Programming Language :: Python :: 3 :: Only",
        "Development Status :: 1 - Planning",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        ],
    long_description = """\
Stamilarity : quantify the statistical similarity of experimental samples
-------------------------------------------------------------------------


    Please refer to https://stamilarity.readthedocs.org.
"""
)

# FIXME: Rajouter la licence
# FIXME: Ajouter un manifest (http://www.diveintopython3.net/packaging.html#divingin)
