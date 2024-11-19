<a name="readme-top"></a>

# FilteringTechniques

Using Prediction Filters, Bandpass Filters (Ormsby &amp; Butterworth), and Match Filters on data.

## About The Project

`FilteringTechniques` is a set of Python functions that can be used to perform various different kinds of filtering methods on geophysical (or other forms of) data.

The code includes filters such as Prediction, Bandpass, Ormsby, Butterworth, and Match filters.

The goal here was to use some sample data, including a simulated wavelet, and some synthetic seismic-measured data, to then use some of the filtering techniques and analyze their capability to extract the reflectivity series.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### **Root Files**

```sh
.
├── README.md          # This file
├── .gitignore         # Files to ignore during our git process
├── Makefile           # Commands for initializing
├── requirements.txt   # Lists the Python package dependencies
├── rawdata.txt        # Synthetic Seismic Data
├── waveletData.txt    # Synthetic Wavelet Data
├── FilteringTechniques.py # The file storing all the filtering functions I created
└── GenerateFigures.py # The file to generate figures and utilize the sample data and the created functions

```

The root directory includes important files for building and managing the project.

- Makefile: Simplifies common tasks such as building the package or installing requirements.

- requirements.txt: Specifies the required Python packages.

- README.md: Provides an overview and documentation of the project.

The other files are all core files including the FilteringTechniques.py which holds the actual functions and filters I created, and then the GenerateFigures.py that builds figures and utilizes those functions

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Python 3.12 or higher

Git (for cloning the repository)

### Installation

1. Clone the repository

   ```sh
   git clone https://github.com/Feromond/FilteringTechniques.git
   cd FilteringTechniques
   ```

2. Install the required packages and setup venv

   ```make
   make init
   ```

3. Now you should be able to run the `GenerateFigures.py` and see some figures appear, indicating it all worked.

  <p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Jacob Mish - [Portfolio](https://jacobmish.com) - [LinkedIn](https://www.linkedin.com/in/jacob-mish-25915722a/) - JacobPMish@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>
