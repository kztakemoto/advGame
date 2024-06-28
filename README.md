# advGame

This repository contains data and code used

## Terms of use

MIT licensed. Happy if you cite our papers when utilizing the codes:

Takemoto K (2024)

## Requirements
* Python 3.11
```
pip install -r requirements.txt
```

## Usage
### Run Prisoner's Dilemma Game in Model Networks
in Erdos-Renyi networks
```
python run_game.py --network ER
```
Note that $N=t_{\max}=1000$, $\langle k \rangle = 8$, $b=1.4$, and $\epsilon=0.2$ are in default configuration (see `run_game.py` for details).

To specify the network model, use the following arguments:
* ``--network BA``: Barabasi-Albert model
* ``--network WS``: Watts-Strogatz model

To specify the link weight adjustment method, add the following arguments:
* ``--random``: random attacks
* ``--adaptive``: Li et al.'s method


### Run Prisoner's Dilemma Game in Real-World Networks
in Facebook network
```
python run_game.py --network facebook_combined
```

To specify the network, use the following arguments:
* ``--network soc-advogato``: Advogato network
* ``--network soc-anybeat``: AnyBeat network
* ``--network soc-hamsterster``: HAMSTERster network

### Run with different b and $\epsilon$
```
bash run.sh
```

### Plot the Results
Line plots of $\rho$ versus $b$ for different $\epsilon$ (Figs. 1 and 4)
```
python plot_coopratio_vs_payoffb_wrt_eps.py --network ER
```

Line plots of $\rho$ versus $\epsilon$ (e.g., Figs. 2 and 3)
```
python plot_coopratio_vs_eps.py --network ER
```

