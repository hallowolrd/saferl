# Stats Appendix — RQ1 & RQ2

- n per cell: 5 seeds (42, 123, 456, 789, 2024)
- Cost/day: CNY, lower better
- Violation rate: %, lower better
- FCSD: fuzzy constraint satisfaction degree, higher better
- CI: 95% t-CI, df = 4
- Test: Welch's two-sample t-test (unequal variance)
- Effect size: Cohen's d with pooled SD

## RQ1 descriptive (mean ± 95% CI, n=5)

| Algorithm | Scenario | n | Cost/day | Violation % | FCSD | Train s |
|---|---|---:|---:|---:|---:|---:|
| SAC | S1_grid_normal | 5 | 14,131 ± 2,753 | 69.88 | 0.717 | 2,366 |
| PPO | S1_grid_normal | 5 | 16,728 ± 13,924 | 88.84 | 0.632 | 433 |
| PPO-Lag | S1_grid_normal | 5 | 21,212 ± 3,232 | 5.45 | 0.980 | 484 |
| CPO | S1_grid_normal | 5 | 15,808 ± 677 | 0.00 | 1.000 | 621 |
| SL | S1_grid_normal | 5 | 32,728 ± 9,522 | 0.00 | 0.991 | 2,318 |
| HFG-SAC (ours) | S1_grid_normal | 5 | 14,997 ± 1,047 | 0.00 | 0.940 | 4,203 |
| SAC | S2_grid_extreme | 5 | 17,765 ± 2,104 | 89.52 | 0.636 | 1,623 |
| PPO | S2_grid_extreme | 5 | 20,436 ± 11,620 | 82.44 | 0.634 | 445 |
| PPO-Lag | S2_grid_extreme | 5 | 40,971 ± 13,558 | 6.49 | 0.976 | 489 |
| CPO | S2_grid_extreme | 5 | 29,260 ± 1,976 | 0.00 | 0.999 | 1,076 |
| SL | S2_grid_extreme | 5 | 34,896 ± 4,230 | 0.00 | 0.959 | 2,356 |
| HFG-SAC (ours) | S2_grid_extreme | 5 | 23,442 ± 1,867 | 0.00 | 0.900 | 4,251 |
| SAC | S3_island_normal | 5 | 6,461 ± 1,078 | 97.98 | 0.784 | 1,792 |
| PPO | S3_island_normal | 5 | 5,558 ± 2,189 | 98.93 | 0.775 | 465 |
| PPO-Lag | S3_island_normal | 5 | 12,082 ± 4,197 | 30.51 | 0.926 | 325 |
| CPO | S3_island_normal | 5 | 10,963 ± 360 | 1.22 | 0.997 | 324 |
| SL | S3_island_normal | 5 | 8,894 ± 2,787 | 12.98 | 0.959 | 1,337 |
| HFG-SAC (ours) | S3_island_normal | 5 | 9,097 ± 1,916 | 0.60 | 0.992 | 3,088 |
| SAC | S4_island_extreme | 5 | 14,286 ± 2,782 | 95.09 | 0.767 | 1,405 |
| PPO | S4_island_extreme | 5 | 13,347 ± 3,182 | 78.69 | 0.834 | 411 |
| PPO-Lag | S4_island_extreme | 5 | 16,683 ± 1,113 | 40.86 | 0.912 | 480 |
| CPO | S4_island_extreme | 5 | 17,632 ± 509 | 0.06 | 1.000 | 835 |
| SL | S4_island_extreme | 5 | 13,258 ± 3,987 | 34.58 | 0.890 | 2,442 |
| HFG-SAC (ours) | S4_island_extreme | 5 | 14,925 ± 2,639 | 0.62 | 0.993 | 5,664 |

## RQ2 descriptive (mean ± 95% CI, n=5)

| Algorithm | Scenario | n | Cost/day | Violation % | FCSD | Train s |
|---|---|---:|---:|---:|---:|---:|
| SAC | extremity_0.8 | 5 | 11,192 ± 1,953 | 94.35 | 0.789 | 1,890 |
| PPO-Lag | extremity_0.8 | 5 | 12,177 ± 3,112 | 24.55 | 0.948 | 288 |
| CPO | extremity_0.8 | 5 | 14,252 ± 1,166 | 4.49 | 0.990 | 291 |
| SL | extremity_0.8 | 5 | 9,874 ± 2,719 | 42.80 | 0.863 | 895 |
| HFG-SAC (ours) | extremity_0.8 | 5 | 12,074 ± 616 | 0.36 | 0.993 | 1,705 |
| SAC | extremity_1.0 | 5 | 13,557 ± 2,921 | 98.39 | 0.789 | 1,871 |
| PPO-Lag | extremity_1.0 | 5 | 13,608 ± 2,708 | 40.33 | 0.914 | 285 |
| CPO | extremity_1.0 | 5 | 13,444 ± 2,587 | 9.02 | 0.975 | 300 |
| SL | extremity_1.0 | 5 | 10,620 ± 3,597 | 30.98 | 0.901 | 900 |
| HFG-SAC (ours) | extremity_1.0 | 5 | 12,363 ± 460 | 0.45 | 0.993 | 1,701 |
| SAC | extremity_1.1 | 5 | 14,809 ± 2,373 | 98.69 | 0.764 | 1,452 |
| PPO-Lag | extremity_1.1 | 5 | 12,899 ± 3,045 | 51.90 | 0.882 | 300 |
| CPO | extremity_1.1 | 5 | 14,063 ± 2,789 | 9.82 | 0.973 | 287 |
| SL | extremity_1.1 | 5 | 9,778 ± 2,280 | 46.28 | 0.854 | 897 |
| HFG-SAC (ours) | extremity_1.1 | 5 | 12,875 ± 633 | 0.42 | 0.993 | 1,702 |
| SAC | extremity_1.2 | 5 | 15,052 ± 1,893 | 96.93 | 0.768 | 1,742 |
| PPO-Lag | extremity_1.2 | 5 | 13,502 ± 3,263 | 47.89 | 0.899 | 309 |
| CPO | extremity_1.2 | 5 | 14,620 ± 1,895 | 2.32 | 0.992 | 289 |
| SL | extremity_1.2 | 5 | 9,984 ± 3,400 | 49.40 | 0.841 | 901 |
| HFG-SAC (ours) | extremity_1.2 | 5 | 12,799 ± 649 | 0.98 | 0.990 | 1,696 |
| SAC | extremity_1.3 | 5 | 14,723 ± 2,400 | 98.54 | 0.767 | 2,261 |
| PPO-Lag | extremity_1.3 | 5 | 14,508 ± 3,837 | 56.19 | 0.881 | 287 |
| CPO | extremity_1.3 | 5 | 15,518 ± 1,301 | 3.04 | 0.991 | 291 |
| SL | extremity_1.3 | 5 | 10,522 ± 1,154 | 49.17 | 0.841 | 902 |
| HFG-SAC (ours) | extremity_1.3 | 5 | 13,195 ± 489 | 1.22 | 0.989 | 1,757 |
| SAC | extremity_1.4 | 5 | 14,074 ± 1,949 | 98.48 | 0.778 | 861 |
| PPO-Lag | extremity_1.4 | 5 | 13,248 ± 6,129 | 56.93 | 0.844 | 285 |
| CPO | extremity_1.4 | 5 | 15,953 ± 1,034 | 5.48 | 0.988 | 302 |
| SL | extremity_1.4 | 5 | 11,554 ± 3,295 | 48.30 | 0.843 | 908 |
| HFG-SAC (ours) | extremity_1.4 | 5 | 14,121 ± 1,180 | 1.61 | 0.988 | 2,091 |
| SAC | extremity_1.5 | 5 | 14,669 ± 2,053 | 99.05 | 0.765 | 852 |
| PPO-Lag | extremity_1.5 | 5 | 10,948 ± 7,820 | 86.93 | 0.753 | 286 |
| CPO | extremity_1.5 | 5 | 16,330 ± 904 | 13.39 | 0.970 | 310 |
| SL | extremity_1.5 | 5 | 10,501 ± 2,404 | 66.73 | 0.779 | 900 |
| HFG-SAC (ours) | extremity_1.5 | 5 | 14,817 ± 630 | 3.15 | 0.983 | 1,494 |

## RQ1 inferential: HFG-SAC vs each baseline (cost/day)

Welch's two-sided t-test, n=5 vs n=5. Cohen's d = (HFG − baseline)/pooled SD.
Positive Δ means HFG is costlier; positive d = HFG costs more.

| Scenario | Baseline | HFG cost | Base cost | Δ | t | p | Cohen's d |
|---|---|---:|---:|---:|---:|---:|---:|
| S1_grid_normal | SAC | 14,997 | 14,131 | +865 | +0.82 | 0.4508 | +0.52 |
| S1_grid_normal | PPO | 14,997 | 16,728 | -1,732 | -0.34 | 0.7478 | -0.22 |
| S1_grid_normal | PPO-Lag | 14,997 | 21,212 | -6,215 | -5.08 | 0.0042 | -3.21 |
| S1_grid_normal | CPO | 14,997 | 15,808 | -811 | -1.81 | 0.1149 | -1.14 |
| S1_grid_normal | SL | 14,997 | 32,728 | -17,731 | -5.14 | 0.0064 | -3.25 |
| S2_grid_extreme | SAC | 23,442 | 17,765 | +5,677 | +5.60 | 0.0005 | +3.54 |
| S2_grid_extreme | PPO | 23,442 | 20,436 | +3,006 | +0.71 | 0.5155 | +0.45 |
| S2_grid_extreme | PPO-Lag | 23,442 | 40,971 | -17,529 | -3.56 | 0.0222 | -2.25 |
| S2_grid_extreme | CPO | 23,442 | 29,260 | -5,818 | -5.94 | 0.0003 | -3.76 |
| S2_grid_extreme | SL | 23,442 | 34,896 | -11,454 | -6.88 | 0.0007 | -4.35 |
| S3_island_normal | SAC | 9,097 | 6,461 | +2,636 | +3.33 | 0.0147 | +2.11 |
| S3_island_normal | PPO | 9,097 | 5,558 | +3,538 | +3.38 | 0.0099 | +2.14 |
| S3_island_normal | PPO-Lag | 9,097 | 12,082 | -2,985 | -1.80 | 0.1261 | -1.14 |
| S3_island_normal | CPO | 9,097 | 10,963 | -1,866 | -2.66 | 0.0527 | -1.68 |
| S3_island_normal | SL | 9,097 | 8,894 | +203 | +0.17 | 0.8725 | +0.11 |
| S4_island_extreme | SAC | 14,925 | 14,286 | +640 | +0.46 | 0.6556 | +0.29 |
| S4_island_extreme | PPO | 14,925 | 13,347 | +1,578 | +1.06 | 0.3213 | +0.67 |
| S4_island_extreme | PPO-Lag | 14,925 | 16,683 | -1,758 | -1.70 | 0.1450 | -1.08 |
| S4_island_extreme | CPO | 14,925 | 17,632 | -2,707 | -2.80 | 0.0451 | -1.77 |
| S4_island_extreme | SL | 14,925 | 13,258 | +1,668 | +0.97 | 0.3654 | +0.61 |

## RQ2 inferential: HFG-SAC vs each baseline (cost/day)

Welch's two-sided t-test, n=5 vs n=5. Cohen's d = (HFG − baseline)/pooled SD.
Positive Δ means HFG is costlier; positive d = HFG costs more.

| Scenario | Baseline | HFG cost | Base cost | Δ | t | p | Cohen's d |
|---|---|---:|---:|---:|---:|---:|---:|
| extremity_0.8 | SAC | 12,074 | 11,192 | +881 | +1.19 | 0.2880 | +0.76 |
| extremity_0.8 | PPO-Lag | 12,074 | 12,177 | -104 | -0.09 | 0.9318 | -0.06 |
| extremity_0.8 | CPO | 12,074 | 14,252 | -2,178 | -4.59 | 0.0036 | -2.90 |
| extremity_0.8 | SL | 12,074 | 9,874 | +2,200 | +2.19 | 0.0873 | +1.39 |
| extremity_1.0 | SAC | 12,363 | 13,557 | -1,194 | -1.12 | 0.3223 | -0.71 |
| extremity_1.0 | PPO-Lag | 12,363 | 13,608 | -1,245 | -1.26 | 0.2731 | -0.80 |
| extremity_1.0 | CPO | 12,363 | 13,444 | -1,081 | -1.14 | 0.3136 | -0.72 |
| extremity_1.0 | SL | 12,363 | 10,620 | +1,743 | +1.33 | 0.2509 | +0.84 |
| extremity_1.1 | SAC | 12,875 | 14,809 | -1,934 | -2.19 | 0.0855 | -1.38 |
| extremity_1.1 | PPO-Lag | 12,875 | 12,899 | -24 | -0.02 | 0.9839 | -0.01 |
| extremity_1.1 | CPO | 12,875 | 14,063 | -1,188 | -1.15 | 0.3074 | -0.73 |
| extremity_1.1 | SL | 12,875 | 9,778 | +3,097 | +3.63 | 0.0173 | +2.30 |
| extremity_1.2 | SAC | 12,799 | 15,052 | -2,253 | -3.13 | 0.0266 | -1.98 |
| extremity_1.2 | PPO-Lag | 12,799 | 13,502 | -703 | -0.59 | 0.5869 | -0.37 |
| extremity_1.2 | CPO | 12,799 | 14,620 | -1,821 | -2.52 | 0.0536 | -1.60 |
| extremity_1.2 | SL | 12,799 | 9,984 | +2,815 | +2.26 | 0.0823 | +1.43 |
| extremity_1.3 | SAC | 13,195 | 14,723 | -1,529 | -1.73 | 0.1526 | -1.10 |
| extremity_1.3 | PPO-Lag | 13,195 | 14,508 | -1,313 | -0.94 | 0.3978 | -0.60 |
| extremity_1.3 | CPO | 13,195 | 15,518 | -2,324 | -4.64 | 0.0053 | -2.94 |
| extremity_1.3 | SL | 13,195 | 10,522 | +2,673 | +5.92 | 0.0015 | +3.74 |
| extremity_1.4 | SAC | 14,121 | 14,074 | +46 | +0.06 | 0.9569 | +0.04 |
| extremity_1.4 | PPO-Lag | 14,121 | 13,248 | +873 | +0.39 | 0.7163 | +0.25 |
| extremity_1.4 | CPO | 14,121 | 15,953 | -1,833 | -3.24 | 0.0121 | -2.05 |
| extremity_1.4 | SL | 14,121 | 11,554 | +2,567 | +2.04 | 0.0972 | +1.29 |
| extremity_1.5 | SAC | 14,817 | 14,669 | +149 | +0.19 | 0.8557 | +0.12 |
| extremity_1.5 | PPO-Lag | 14,817 | 10,948 | +3,870 | +1.37 | 0.2418 | +0.87 |
| extremity_1.5 | CPO | 14,817 | 16,330 | -1,512 | -3.81 | 0.0064 | -2.41 |
| extremity_1.5 | SL | 14,817 | 10,501 | +4,316 | +4.82 | 0.0061 | +3.05 |
