# PriceOfJR

This folder contains the code for reproducing the results presented in the paper "The Price of justified representation" (Edith Elkind, Piotr Faliszewski, Ayumi Igarashi, Pasin Manurangsi, Ulrike Schmidt-Kraepelin, and Warut Suksompong, AAAI 2022). 

**Important:** This code builds upon code contributed by Andrzej Kaczmarczyk (link to be added). More specifically, our code uses the files "baseProgram", "distributions", and "isxJRChecker". 

In order to recreate our results, download the above mentioned files and run our script 'main.py' with python3. Due to the large number of ILPs that need to be solved, running the code can take a few hours. Besides the packages stated below, a gurobi license is required (academic license, free trial does not suffice).

Used packages:
- pandas 1.2.4
- numpy 1.20.1
- matplotlib 3.3.4
- gurobipy 9.1.2
- pulp  2.4
- math 
- random
