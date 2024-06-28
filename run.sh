b_values=(1.05 1.15 1.25 1.35 1.45 1.55 1.65 1.75 1.85 1.95)
eps_values=(0.1 0.2 0.0 -0.1 -0.2)

for eps in "${eps_values[@]}"; do
  for b in "${b_values[@]}"; do
    echo ER $b $kave $eps
    python run_Game.py --payoff_b "$b" --eps "$eps" --network ER
  done
done
