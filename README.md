# PyWinEA

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
<br />
<p align="center">
	<img src="https://github.com/FernandoGaGu/pywinEA/blob/master/img/PyWinEA_logo.gif" alt="Logo">
</p>

> Python package with lightweight implementations of genetic algorithms for classification/regression tasks.

## Description

The pywinEA module is a native python implementation of some of the most widely used genetic algorithms. 
This package has been developed on the top of scikit-learn which allows to use any model already implemented. This module aims to provide a good alternative to other feature selection techniques with full scikit-learn compatibility.

**Why evolutionary algorithms?**

One of the first stages in the development of any machine learning model is to filter out redundant and/or irrelevant attributes. However, the complexity of finding the best combination of attributes is most often an NP problem.

Among the most frequent feature selection strategies are embedded methods. These methods combine a heuristic search strategy with a classification/regression model. This is where genetic algorithms come into play. This type of strategy represents one of the best alternatives to address the immense space of search generally reaching good solutions.

## Install

### Dependencies

PyWinEA requires:
- Python (>= 3.6)
- NumPy (>= 1.13.3)
- SciPy (>= 0.19.1)
- Scikit-learn (>= 0.20.0)
- tqdm (>= 4.42.1)
- matplotlib (>= 3.1.3)
- pandas (>= 1.0.1)

```sh
pip install pywinEA
```
It is possible that older versions of the packages listed above may work. However, full compatibility is not guaranteed.

## Usage

Examples of the basic use of the package can be found in the notebooks directory. A diagram of the module structure is also shown below. For more advanced use it is recommended to look at the documentation. 

Additionally by using the classes defined in the interface subpackage it is possible to implement new operators, algorithms, etc. Feel free to add things.

The following is an example of the most basic implementation of a genetic algorithm.
```python
# Basic GA
from pywinEAt.algorithm import GA
from sklearn.naive_bayes import GaussianNB  # Fitness function

# Data loading and processing...

POPULATION_SIZE = 50
GENERATIONS = 200
FITNESS = GaussianNB()
CV = 5
ANNHILATION = 0.1
ELITISM = 0.2
MUTATION = 0.1

ga_basic = GA(
    population_size=POPULATION_SIZE, generations=GENERATIONS, cv=CV,
    fitness=FITNESS, annihilation=ANNHILATION, elitism=ELITISM, 
    mutation_rate=MUTATION, positive_class=1, id="BasicGA"
)

ga_basic.set_features(feature_names)   #Selection of the feature names

# Fit algorithm 
ga_basic.fit(x_data, y_data) 

# Get the names of the most relevant features
ga_basic.best_features 
```
<br />
<p align="center">
	<img src="https://github.com/FernandoGaGu/pywinEA/blob/master/img/basic-example.gif" alt="Example">
</p>

This type of algorithm usually works well, however we may be interested in maximizing two objectives, for example the performance of the classifier (maximization) and the number of characteristics (minimization). In this case the multi-target algorithms (NSGA2 and SPEA2) are the best alternative.

```python
from pywinEAt.algorithm import NSGA2
from sklearn.naive_bayes import GaussianNB  # Fitness function

# Data loading and processing...

POPULATION_SIZE = 50
GENERATIONS = 200
FITNESS = GaussianNB()
CV = 5
ANNHILATION = 0.1
ELITISM = 0.2
MUTATION = 0.1

nsga = NSGA2(
    population_size=POPULATION_SIZE, generations=GENERATIONS, 
    fitness=mono_objective, mutation_rate=ELITISM, 
    optimize_features=True, positive_class=1, id="NSGA2"
)

nsga.set_features(feature_names)   #Selection of the feature names

# Fit algorithm 
nsga.fit(x_data, y_data) 

# Get the names of the most relevant features
nsga.best_features 
```
The result of the multi-objective algorithms is a non-dominant front of solutions. For example:

<br />
<p align="center">
	<img src="https://github.com/FernandoGaGu/pywinEA/blob/master/img/Pareto-front-example.png" alt="Example">
</p>

(Complete examples: [notebooks](https://github.com/FernandoGaGu/pywinEA/tree/master/notebooks/))

## Module structure

<br />
<p align="center">
	<img src="https://github.com/FernandoGaGu/pywinEA/blob/master/img/PyWinEAStructure.png" alt="Structure">
</p>


## Notes

The package is still in testing, it is possible to find some unexpected errors. Any problem 👉  <a href="https://github.com/FernandoGaGu/pywinEA/issues"> issues </a>



## License

[MIT](LICENSE) © 

[contributors-shield]: https://img.shields.io/github/contributors/FernandoGaGu/pywinEA.svg?style=flat-square
[contributors-url]: https://github.com/FernandoGaGu/pywinEA/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/FernandoGaGu/pywinEA.svg?style=flat-square
[forks-url]: https://github.com/FernandoGaGu/pywinEA/network/members
[stars-shield]: https://img.shields.io/github/stars/FernandoGaGu/pywinEA.svg?style=flat-square
[stars-url]: https://github.com/FernandoGaGu/pywinEA/stargazers
[issues-shield]: https://img.shields.io/github/issues/FernandoGaGu/pywinEA.svg?style=flat-square
[issues-url]: https://github.com/FernandoGaGu/pywinEA/issues
[license-shield]: https://img.shields.io/github/license/FernandoGaGu/pywinEA.svg?style=flat-square
[license-url]: https://github.com/FernandoGaGu/pywinEA/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/GarciaGu-Fernando
[product-screenshot]: img/PyWinEAlogo.png
