# Degenerate Preconditioned Proximal Point Algorithms: Example Code

This repository contains the experimental source code for computing an optimal solution to a regularized Markowitz portfolio optimization problem with transaction penalties with different operator splitting methods used for numerical comparisons in:
* K. Bredies, E. Chenchene, D. Lorenz, E. Naldi. Degenerate Preconditioned Proximal Point Algorithms. [*SIAM J. Optim.*, 32(3):2376—2401](https://doi.org/10.1137/21M1448112), 2022. [ArXiv preprint](https://arxiv.org/abs/2109.11481)

To reproduce the results of the numerical experiment in Section 3.2, run:
```bash
python3 comparisons.py
```
Convergence plots are saved in the file `test2.pdf`.

If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@article{bcln2022,
	author = {Bredies, Kristian and Chenchene, Enis and Lorenz, Dirk A. and Naldi, Emanuele},
	title = {Degenerate Preconditioned Proximal Point Algorithms},
	journal = {SIAM Journal on Optimization},
	volume = {32},
	number = {3},
	pages = {2376--2401},
	year = {2022}
}
```
## Acknowledgments

* | ![](<euflag.png>) | This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no. 861137.  |
  |----------|----------|
* The data included in this repository has been downloaded from https://stanford.edu/class/engr108/data/ [Accessed March 27, 2021]. Used with permission.

## License
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
