# AlgEvo

Efficient (almost) single file implementations of evolutionary algorithms.

## Implemented Algorithms

- [Particle Swarm Optimization](https://www.cs.uoi.gr/~kostasp/papers/C26.pdf) (UPSO)
- [Gradient-Assisted Particle Swarm Optimization for Constrained Optimization](http://costashatz.github.io/files/LION17.pdf) (UPSO-QP)
- [Particle Swarm Optimization with Penalties for Constrained Optimization](https://www.cs.cinvestav.mx/~constraint/papers/eisci.pdf) (UPSO-Penalty)
- [Particle Swarm Optimization with Gradient Repair Scheme](https://www.sciencedirect.com/science/article/abs/pii/S030505480500050X) (UPSO-Grad)
- [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) (DE)
- [MAP-Elites](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8000667) (CVT-MAP-Elites)
- [Uncertain MAP-Elites](https://arxiv.org/pdf/2302.00463.pdf) (preliminary: works only for noisy objective function, not noisy features)
- [improved Cross Entropy Method](https://martius-lab.github.io/iCEM/) (iCEM)

## Compilation/Installation

### Dependencies

- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [TBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html)
- [ProxQP](https://github.com/Simple-Robotics/proxsuite) *(optional for UPSO-QP)*

### Compilation/Installation

- `./waf configure [--prefix=PATH_TO_INSTALL]`
- `./waf`
- `[sudo] ./waf install`

## Running the Examples

There are numerous examples under `src/examples`. If the compilation procedure has successfully completely, you can run them by `./build/example_name`.

## Citing AlgEvo

If you use `AlgEvo` in a scientific publication, please use the following citation ([pdf](http://costashatz.github.io/files/LION17.pdf)):

```bibtex
@inproceedings{chatzilygeroudis2023lion,
    title={Fast and Robust Constrained Optimization via Evolutionary and Quadratic Programming},
    author={Chatzilygeroudis, Konstantinos and Vrahatis, Michael},
    year={2023},
    booktitle={The 17th Learning and Intelligent Optimization Conference (LION)}
}
```

## Acknowledgments

This work was supported by the [Hellenic Foundation for Research and Innovation](https://www.elidek.gr/en/homepage/) (H.F.R.I.) under the "3rd Call for H.F.R.I. Research Projects to support Post-Doctoral Researchers" (Project Acronym: NOSALRO, Project Number: 7541).

<p align="center">
<img src="https://www.elidek.gr/wp-content/themes/elidek/images/elidek_logo_en.png" alt="logo_elidek"/>
<p/>

<!-- <center>
<img src="https://nosalro.github.io/images/logo_elidek.png" alt="logo_elidek" width="50%"/>
</center> -->

This work was conducted within the [Computational Intelligence Lab](http://cilab.math.upatras.gr/) (CILab), Department of Mathematics, University of Patras, Greece.

<p align="center">
<img src="https://nosalro.github.io/images/logo_cilab.jpg" alt="logo_cilab" width="50%"/>
<img src="https://www.upatras.gr/wp-content/uploads/up_2017_logo_en.png" alt="logo_cilab" width="50%"/>
</p>

## License

[BSD 2-Clause "Simplified" License](https://opensource.org/license/bsd-2-clause/)

