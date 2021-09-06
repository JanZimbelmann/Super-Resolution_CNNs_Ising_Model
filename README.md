# Super-Resolution with Convolutional Neural Networks (CNN) for the 2D Ising Model

A CNN is trained on a spin configuration to revert a renormalization group (RG) decimation (learn.py). Once trained it is possible to super-resolve to larger system sizes (recon.py). The spin configuration is created with Metropolis Monte Carlo (MC) simulations and decimated according to a deterministic majority rule(configurations_ising.cpp). The solution of the super-resolution procedure is compared to the solutions of seperate Metropolis MC simulations (solutions_ising.cpp). For the super-resolution procedure, the temperature changes according to the renormalizaton. For the 2D case this cannot be solved analytically and is here solved numerically (numerical_ising.cpp + temperature_regression.py).

This work is a part of my master thesis (Master-Thesis.pdf) and includes the simulations for the 2D case. My thesis describes this method in further detail and covers the theory behind the neural network models used in this project as well as the physics to understand the ising model and renormalizatoin group procedure. The Thesis is also extended for the 1D and 3D super-resolution procedure.

## Background

The Ising model is a nearest neighbor interaction approximation of a magnet with each spin being arranged in a binary state on a lattice. It serves as a very important toy model in computational physics and there are many noticeable problems with it which also apply to more complex models. One example is the critical slowing down in computational simulations of the Ising model at the critical temperature of the phase transition. It is a problem which is originated from a divergent correlation length of the interactions. This is especially crucial when studying larger system sizes, since it would take very long to capture all important states in a more traditional computer simulation, like a Monte Carlo (MC) simulation with a Metropolis algorithm.

When using a renormalization procedure on a Ising spin configuration to decrease the system length to a half, the idea is then to train a neural network with supervised learning to revert the renormalization to its original system size. It is not required for the super resolution to be an exact reversion to the problem, since only the probability distribution is what plays a role when calculating a statistical average of an observable. It is then possible to run a computer simulation with a MC method of the Ising model at small system sizes and rescale their properties to a much larger size. The idea comes from a paper from [Efthymiou, Beach and Melko, which was published on the 31st of January 2019](https://arxiv.org/abs/1810.02372). The here presented code captures the method of this paper, however the master thesis extend the work by the 3D case and gives a much greater analysis on the boundaries of this method by analyzing finite system size problems.

## Order of code execution

There are multiple parts of code for this methodology which build up on the results of the previous code. Now all the pieces are presented in the order of execution. It is noted that simulations for multiple temperature points are evaluated. However learn.py and recon.py only target a single data point which is targeted with a argument parser. For the thesis a script is run to iterate over all desired temperature points. This is calculated with the help of a computer cluster. All other code is targeting all temperature points in a single execution. It is also noted that no visualization of the results are presented in this repository. The results can only be viewed in the resulting .csv file. However the results are also to be seen within the  Master-Thesis.pdf.

### numerical_ising.cpp

A super-resolution step changes the temperature of the previously simulated spin configuration. This transformation cannot be solved analytically and therefore needs to be solve numerically. This C++ code simulates the Ising model at some original system size. It also simulated the Ising model at twice the original system size and decimates it to the original system size according to the majority rule. This is done for a temperature range. The magnetization is calculated and stored.

### temperature_regression.cpp

The previously stored magnetization of the original and decimated spin configuration is loaded and a polynomial regression is calculated. Hereby a numerical regression for the temperature transformation is found and a list of initial temperature points is transformed a certain amount of times. The intial temperatures and the reiterating transformation of it is stored in a .csv file.

### configurations_ising.cpp

The previously intial temperature list with the numerical solutions for the temperature transformation is loaded and ising spin configurations are generated for all of those temperatures. The Ising spin configurations are also decimated according to the deterministic majority rule to half the original system size. The original and decimated spin configurations are stored in a .csv file.

### learn.py

A CNN model is prepared and trained to math the decimation procedure to match the original spin configuration distribution. The network hereby learns to revert from a system length of L/2 to L. The weights and biases are stored.

### recon.py

An equal CNN model is prepared and the weights and biases are loaded from the previous code. Now The initial spin configuration is enlarged by twice the system size with each super resolution step: L -> 2L -> 4L -> ... The solution for the observables are stored.

### solutions_ising.cpp

To compare the previously obtained solutions, the Ising model is simulated once again with the Metropolis Monte Carlo method for the previously calculated temperatures. Now it is also simulated for the larger system sizes and not only for the original system size. This code can be run at the same time like the 'configurations_ising.cpp' but is not necessary for the CNN super-resolution procedure.

## Contribution

The code is written by Jan Zimbelmann
