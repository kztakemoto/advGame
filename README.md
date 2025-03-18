# advGame

This repository contains data and code used in [*Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks*](https://doi.org/10.1016/j.physa.2024.130214)

## Terms of use

This project is MIT licensed. We appreciate citations of our paper when using this code:

Takemoto K (2024) **Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks.** Physica A 655, 130214.

## Requirements
* Python 3.11

Install dependencies:
```
pip install -r requirements.txt
```

## Usage
### Run Prisoner's Dilemma Game in Model Networks
For Erdős-Rényi networks:
```
python run_game.py --network ER
```
Default configuration: $N=t_{\max}=1000$, $\langle k \rangle = 8$, $b=1.4$, and $\epsilon=0.2$ (see ``run_game.py`` for details).

To specify other network models, use the following arguments:
* ``--network BA``: Barabási-Albert model
* ``--network WS``: Watts-Strogatz model

To specify the link weight adjustment method, add one of these arguments:
* ``--random``: random attacks
* ``--adaptive``: Li et al.'s method


### Run Prisoner's Dilemma Game in Real-World Networks
For Facebook network:
```
python run_game.py --network facebook_combined
```

Other available networks:
* ``--network soc-advogato``: Advogato network
* ``--network soc-anybeat``: AnyBeat network
* ``--network soc-hamsterster``: HAMSTERster network

### Run with different $b$ and $\epsilon$ Values
```
bash run.sh
```

### Plot the Results
Line plots of $\rho$ versus $b$ for different $\epsilon$ (Figs. 1 and 4):
```
python plot_coopratio_vs_payoffb_wrt_eps.py --network ER
```

Line plots of $\rho$ versus $\epsilon$ (Figs. 2 and 3):
```
python plot_coopratio_vs_eps.py --network ER
```

