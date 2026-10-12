# Figure Catalog — RQ1 & RQ2

## fig01_rq1_cost.png
- **Purpose**: show cost/day across 4 scenarios for all 6 algorithms.
- **Variables**: x = scenario (S1–S4), y = mean cost/day (CNY), bars grouped by algorithm; error bars = 95% t-CI (n=5).
- **Key observation**: HFG-SAC (red) is consistently in the middle of the cost band; it is far cheaper than PPO-Lag/SL/CPO in S2 and tied with baselines in S3/S4. SAC/PPO are cheapest only because they violate safety.
- **What it changes**: anchors the claim "HFG-SAC is cost-competitive with safe baselines" rather than "cheapest".
- **Caveat**: PPO's CI in S1/S2 is enormous (σ dominated by one bad seed), so its apparent low cost is unreliable.

## fig02_rq1_violation.png
- **Purpose**: show hard-constraint violation rate with the 5% safety threshold.
- **Variables**: same layout as fig01; y = violation rate (%), red dotted line = 5%.
- **Key observation**: only HFG-SAC, CPO, and SL stay at/near 0% in grid-connected S1/S2; in islanded S3/S4, SL and PPO-Lag climb sharply, while HFG-SAC remains below 1%.
- **What it changes**: the safety win is not a grid-connected-only phenomenon; it holds in islanded extreme mode where SL/PPO-Lag fail.
- **Caveat**: y-axis extends to 120%; the 5% threshold is near zero, making small differences hard to read — a zoomed panel for S3/S4 violations would help.

## fig03_rq1_pareto.png
- **Purpose**: cost–safety Pareto scatter; marker shape = scenario, color = algorithm.
- **Variables**: x = mean cost/day, y = mean violation rate (%).
- **Key observation**: HFG-SAC (red stars) clusters in the lower-left corner across all four marker shapes. CPO is also low-violation but sits 5–8k to the right (more expensive). SAC/PPO occupy the upper-right (cheap but unsafe).
- **What it changes**: this is the single most decisive figure — visually demonstrates joint dominance.
- **Caveat**: no CI on the scatter points; the Pareto argument assumes means are unbiased.

## fig04_rq2_curves.png
- **Purpose**: cost (left) and violation rate (right) as functions of demand extremity λ.
- **Variables**: x = λ (0.8–1.5), y = metric; shaded bands = 95% CI.
- **Key observation (left)**: HFG-SAC cost grows smoothly; SL is cheaper but unsafe; CPO gets more expensive at high λ.
- **Key observation (right)**: HFG-SAC violation curve stays near 0% across the whole λ sweep and only reaches 3% at λ=1.5. All other baselines cross 5% by λ=0.8 and climb to 13–87%.
- **What it changes**: directly answers "does safety degrade gracefully?" — yes, monotonically and below threshold.
- **Caveat**: PPO-Lag CI band at λ=1.5 is enormous (one safe seed + four unsafe seeds), so its mean is misleading; report seed-level in the appendix.

## fig05_rq2_fcsd_drift.png
- **Purpose**: contrast FCSD at mild (λ=0.8) vs extreme (λ=1.5) for each algorithm.
- **Variables**: grouped bars; numbers = mean FCSD; Δ annotated below.
- **Key observation**: SAC loses ~0.023 FCSD; PPO-Lag collapses from 0.95 to 0.75; CPO drifts 0.99 → 0.97; HFG-SAC drifts only 0.993 → 0.983 (Δ = 0.010), the smallest degradation of any method.
- **What it changes**: shows that the fuzzy constraint satisfaction — not just hard violation — is preserved under tail demand.
- **Caveat**: FCSD differences are small (0.01–0.02) in absolute terms; whether this matters operationally depends on the domain threshold, not on statistical significance.

## fig06_rq1_train_cost.png
- **Purpose**: training dynamics — episode cost over 200 episodes across 4 scenarios.
- **Variables**: x = episode (10-episode moving avg), y = mean episode cost (CNY); shaded band = ±1 std over 5 seeds.
- **Key observation**: HFG-SAC (thick red) converges smoothly to ~100–180k episode cost in all scenarios. SAC/PPO also converge but at lower cost because they violate constraints. PPO-Lag/SL/CPO show much higher variance and in S3/S4 never settle.
- **What it changes**: demonstrates HFG-SAC's training stability — no oscillation, no divergence, monotonic improvement.
- **Caveat**: absolute episode cost is ~7× the daily cost (24 episodes/day), so the y-axis looks large; this is expected.

## fig07_rq1_train_violation.png
- **Purpose**: training dynamics — episode violation rate over 200 episodes.
- **Variables**: x = episode, y = mean episode violation rate (%), 10-episode moving avg.
- **Key observation**: **HFG-SAC drops to ~0% within the first 20 episodes and stays there across ALL 4 scenarios**. SAC/PPO stay at 70–100% throughout (they never learn safety). PPO-Lag reaches 0% only in S1/S2; in S3/S4 it stays at 20–40%. SL fluctuates between 0–40% in S3/S4. CPO gradually decreases but remains at 20–30% in S3/S4.
- **What it changes**: this is the strongest visual evidence for the paper — HFG-SAC is the only method that learns to be safe from the very start of training, and stays safe without the oscillation seen in PPO-Lag/SL.
- **Caveat**: y-axis includes negative values due to moving-average smoothing of zero-violation episodes; this is a smoothing artifact, not a real metric.

## fig08_rq1_train_fcsd.png
- **Purpose**: training dynamics — episode FCSD over 200 episodes.
- **Variables**: x = episode, y = mean episode FCSD, 10-episode moving avg.
- **Key observation**: HFG-SAC starts at 0.83 (S1/S2) or 0.96 (S3/S4) and climbs to 0.94–0.99 by episode 50, then stays there. SAC/PPO are stuck at 0.6–0.75 throughout. PPO-Lag/SL/CPO reach 0.95–0.99 only in S1/S2; in S3/S4 they plateau at 0.85–0.95.
- **What it changes**: shows that HFG-SAC's fuzzy constraint satisfaction is not just a final-evaluation artifact — it converges early and remains stable.
- **Caveat**: HFG-SAC's initial FCSD in S1/S2 (0.83) is lower than SL/PPO-Lag (0.95), but it quickly overtakes them by episode 30–50.
