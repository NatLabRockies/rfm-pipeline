# Coefficient Matrix Column Order

Canonical column order for the 132-feature reduced-form coefficient matrix (`artifacts/final_model/coefficient_matrix_raw_scale.csv` and `coefficient_matrix_standardized.csv`). Each row pairs the column position with its feature name, transformation, base input(s), units, sample range, and the feature's contribution metrics.

**Total columns:** 245. **Source of truth for column order:** `artifacts/final_model/final_support_features.csv` (field `final_support_position`).

| Idx | Feature name | Transform | Base input 1 | Unit | Min | Max | Pathway | n_out≠0 | Δ nRMSE if removed |
| --- | ------------ | --------- | ------------ | ---- | --- | --- | ------- | ------- | ------------------ |
| 0 | `WW.PY sensi multiplier[SludgeToHTL]` | identity | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 3802 | 0.6644 |
| 1 | `WW.PY sensi multiplier[SludgeToHTL]_sqrt` | sqrt | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 3275 | 0.6165 |
| 2 | `WW.PY sensi multiplier[SludgeToHTL]:WW.progress ratios commercial[SludgeToHTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 2600 | 0.2008 |
| 3 | `WW.progress ratios commercial[SludgeToHTL]:WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 3108 | 0.0884 |
| 4 | `AHC.PY sensi multiplier[HEFA]` | identity | `AHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Algal HEFA | 456 | 0.1479 |
| 5 | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | identity | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem | 4103 | 0.1032 |
| 6 | `WW.PY sensi multiplier[ManureToHTL]_sq` | quadratic | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 1426 | 0.0352 |
| 7 | `OHC.PY sensi multiplier[HEFA]` | identity | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 3715 | 0.6760 |
| 8 | `CHC.PY sensi multiplier[Thermochem]_sq` | quadratic | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 4776 | 0.1061 |
| 9 | `CHC.initial indices of Commercial Maturity[Thermochem]` | identity | `CHC.initial indices of Commercial Maturity[Thermochem]` | unitless | 0.1 | 0.7 | Cellulosic Thermochem | 4306 | 0.0727 |
| 10 | `CHC.PY sensi multiplier[Thermochem]:CHC.Mature Industry Rate of Return as PCT[Thermochem]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 6152 | 0.0867 |
| 11 | `WW.progress ratios commercial[SludgeToHTL]` | identity | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 3778 | 0.2404 |
| 12 | `WW.PY sensi multiplier[SludgeToHTL]:WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 4015 | 0.0517 |
| 13 | `WW.Mature Industry Rate of Return as PCT[SludgeToHTL]_sqrt` | sqrt | `WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | %/yr | 5.0 | 15.0 | Sewage Sludge HTL | 3189 | 0.1186 |
| 14 | `CHC.initial indices of Commercial Maturity[Brownfield]:CHC.PY sensi multiplier[Brownfield]` | interaction | `CHC.initial indices of Commercial Maturity[Brownfield]` | unitless | 0.0 | 0.7 | Cellulosic Thermochem (Brownfield) | 5059 | 0.0378 |
| 15 | `AHC.PY sensi multiplier[HTL]` | identity | `AHC.PY sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 563 | 0.0969 |
| 16 | `AHC.PY sensi multiplier[HEFA]_sq` | quadratic | `AHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Algal HEFA | 447 | 0.0515 |
| 17 | `WW.PY sensi multiplier[ManureToHTL]:WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 3052 | 0.0167 |
| 18 | `WW.progress ratios commercial[SludgeToHTL]_sq` | quadratic | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 3486 | 0.2044 |
| 19 | `CHC.initial indices of Commercial Maturity[Brownfield]:CHC.Mature Industry Rate of Return as PCT[Brownfield]` | interaction | `CHC.initial indices of Commercial Maturity[Brownfield]` | unitless | 0.0 | 0.7 | Cellulosic Thermochem (Brownfield) | 5669 | 0.0251 |
| 20 | `CHC.Mature Industry Rate of Return as PCT[Thermochem]_sqrt` | sqrt | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem | 4913 | 0.1445 |
| 21 | `CHC.PY sensi multiplier[Thermochem]:CHC.initial indices of Commercial Maturity[Thermochem]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 5396 | 0.0334 |
| 22 | `AHC.PY sensi multiplier[HTL]_sq` | quadratic | `AHC.PY sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 641 | 0.0367 |
| 23 | `CHC.Mature Industry Rate of Return as PCT[Brownfield]` | identity | `CHC.Mature Industry Rate of Return as PCT[Brownfield]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem (Brownfield) | 1734 | 0.0849 |
| 24 | `CHC.PY sensi multiplier[Thermochem]:CHC.FCI  sensi multiplier[Thermochem]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 5303 | 0.0912 |
| 25 | `CHC.initial indices of Commercial Maturity[Thermochem]_log1p` | log1p | `CHC.initial indices of Commercial Maturity[Thermochem]` | unitless | 0.1 | 0.7 | Cellulosic Thermochem | 3576 | 0.0422 |
| 26 | `WW.initial indices of Commercial Maturity[SludgeToHTL]` | identity | `WW.initial indices of Commercial Maturity[SludgeToHTL]` | Unitless | 0.0 | 0.2 | Sewage Sludge HTL | 2467 | 0.0024 |
| 27 | `WW.progress ratios commercial[SludgeToHTL]:WW.FCI  sensi multiplier[SludgeToHTL]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 2323 | 0.0589 |
| 28 | `SE.Max Jet Investment Hit Rate_log1p` | log1p | `SE.Max Jet Investment Hit Rate` | 1/year | 0.05 | 0.3 | Starch Ethanol-to-Jet | 4232 | 0.0391 |
| 29 | `CHC.initial indices of Commercial Maturity[Brownfield]` | identity | `CHC.initial indices of Commercial Maturity[Brownfield]` | unitless | 0.0 | 0.7 | Cellulosic Thermochem (Brownfield) | 3215 | 0.0378 |
| 30 | `OHC.Retirement Frac[TransEster]` | identity | `OHC.Retirement Frac[TransEster]` | unitless | 0.0 | 0.15 | Oilcrop Transesterification | 1714 | 0.0369 |
| 31 | `CHC.Mature Industry Rate of Return as PCT[Biochem]` | identity | `CHC.Mature Industry Rate of Return as PCT[Biochem]` | %/yr | 5.0 | 15.0 | Cellulosic Biochem | 916 | 0.0267 |
| 32 | `CHC.PY sensi multiplier[Thermochem]:CHC.progress ratios commercial[Thermochem]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 4496 | 0.1342 |
| 33 | `CHC.initial indices of Commercial Maturity[Brownfield]:CHC.FCI  sensi multiplier[Brownfield]` | interaction | `CHC.initial indices of Commercial Maturity[Brownfield]` | unitless | 0.0 | 0.7 | Cellulosic Thermochem (Brownfield) | 3876 | 0.0170 |
| 34 | `CHC.Mature Industry Rate of Return as PCT[Brownfield]_sqrt` | sqrt | `CHC.Mature Industry Rate of Return as PCT[Brownfield]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem (Brownfield) | 3437 | 0.1269 |
| 35 | `SE.Max Jet Investment Hit Rate` | identity | `SE.Max Jet Investment Hit Rate` | 1/year | 0.05 | 0.3 | Starch Ethanol-to-Jet | 4212 | 0.0371 |
| 36 | `CHC.FCI  sensi multiplier[Thermochem]` | identity | `CHC.FCI  sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 3478 | 0.1896 |
| 37 | `CHC.Mature Industry Rate of Return as PCT[Thermochem]:CHC.FCI  sensi multiplier[Thermochem]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem | 4421 | 0.0291 |
| 38 | `CHC.Mature Industry Rate of Return as PCT[Brownfield]:CHC.PY sensi multiplier[Brownfield]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Brownfield]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem (Brownfield) | 3643 | 0.0324 |
| 39 | `CHC.Mature Industry Rate of Return as PCT[Thermochem]:CHC.progress ratios commercial[Thermochem]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem | 4026 | 0.0402 |
| 40 | `WW.PY sensi multiplier[ManureToHTL]` | identity | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 1018 | 0.1259 |
| 41 | `WW.PY sensi multiplier[ManureToHTL]:WW.ORNOOC sensi multiplier[ManureToHTL]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 975 | 0.0125 |
| 42 | `WW.progress ratios commercial[SludgeToHTL]:WW.ORNOOC sensi multiplier[SludgeToHTL]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 2041 | 0.0464 |
| 43 | `WW.PY sensi multiplier[SludgeToHTL]:WW.ORNOOC sensi multiplier[SludgeToHTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 3517 | 0.0365 |
| 44 | `WW.PY sensi multiplier[SludgeToHTL]:WW.FCI  sensi multiplier[SludgeToHTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 2377 | 0.0330 |
| 45 | `CHC.Mature Industry Rate of Return as PCT[Thermochem]:CHC.initial indices of Commercial Maturity[Thermochem]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem | 5715 | 0.0145 |
| 46 | `WW.PY sensi multiplier[ManureToHTL]:WW.progress ratios commercial[ManureToHTL]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 958 | 0.0176 |
| 47 | `OHC.PY sensi multiplier[HEFABrownfield]:OHC.PY sensi multiplier[HEFA]` | interaction | `OHC.PY sensi multiplier[HEFABrownfield]` | unitless | 0.75 | 1.25 | Oilcrop HEFA (Brownfield) | 4473 | 0.1680 |
| 48 | `AHC.initial indices of Commercial Maturity[HTL]` | identity | `AHC.initial indices of Commercial Maturity[HTL]` | unitless | 0.0 | 0.2 | Algal HTL | 2008 | 0.0115 |
| 49 | `OHC.PY sensi multiplier[HEFABrownfield]_log1p` | log1p | `OHC.PY sensi multiplier[HEFABrownfield]` | unitless | 0.75 | 1.25 | Oilcrop HEFA (Brownfield) | 3012 | 0.8258 |
| 50 | `WW.FCI  sensi multiplier[SludgeToHTL]` | identity | `WW.FCI  sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 1721 | 0.0448 |
| 51 | `WW.progress ratios commercial[SludgeToHTL]:AHC.initial indices of Commercial Maturity[HTL]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 1819 | 0.0101 |
| 52 | `CHC.initial indices of Commercial Maturity[Thermochem]:CHC.FCI  sensi multiplier[Thermochem]` | interaction | `CHC.initial indices of Commercial Maturity[Thermochem]` | unitless | 0.1 | 0.7 | Cellulosic Thermochem | 3721 | 0.0119 |
| 53 | `WW.initial indices of Commercial Maturity[SludgeToHTL]_log1p` | log1p | `WW.initial indices of Commercial Maturity[SludgeToHTL]` | Unitless | 0.0 | 0.2 | Sewage Sludge HTL | 2272 | 0.0028 |
| 54 | `CHC.PY sensi multiplier[Brownfield]` | identity | `CHC.PY sensi multiplier[Brownfield]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem (Brownfield) | 2560 | 0.1468 |
| 55 | `CHC.Mature Industry Rate of Return as PCT[Biochem]:CHC.FCI  sensi multiplier[Biochem]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Biochem]` | %/yr | 5.0 | 15.0 | Cellulosic Biochem | 965 | 0.0144 |
| 56 | `CHC.Mature Industry Rate of Return as PCT[Biochem]:CHC.PY sensi multiplier[Biochem]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Biochem]` | %/yr | 5.0 | 15.0 | Cellulosic Biochem | 449 | 0.0144 |
| 57 | `WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | identity | `WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | %/yr | 5.0 | 15.0 | Sewage Sludge HTL | 2440 | 0.0758 |
| 58 | `CHC.PY sensi multiplier[Thermochem]:WW.progress ratios commercial[SludgeToHTL]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 2262 | 0.0453 |
| 59 | `WW.Mature Industry Rate of Return as PCT[SludgeToHTL]:WW.FCI  sensi multiplier[SludgeToHTL]` | interaction | `WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | %/yr | 5.0 | 15.0 | Sewage Sludge HTL | 4159 | 0.0233 |
| 60 | `WW.PY sensi multiplier[ManureToHTL]:WW.initial indices of Commercial Maturity[ManureToHTL]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 997 | 0.0032 |
| 61 | `CHC.PY sensi multiplier[Thermochem]` | identity | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 5285 | 0.3403 |
| 62 | `WW.PY sensi multiplier[SludgeToHTL]:CHC.PY sensi multiplier[Thermochem]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 4985 | 0.0463 |
| 63 | `CHC.FCI  sensi multiplier[Thermochem]_log1p` | log1p | `CHC.FCI  sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 3106 | 0.1720 |
| 64 | `CHC.Mature Industry Rate of Return as PCT[Brownfield]:CHC.FCI  sensi multiplier[Brownfield]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Brownfield]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem (Brownfield) | 2282 | 0.0133 |
| 65 | `CHC.initial indices of Commercial Maturity[Thermochem]:CHC.progress ratios commercial[Thermochem]` | interaction | `CHC.initial indices of Commercial Maturity[Thermochem]` | unitless | 0.1 | 0.7 | Cellulosic Thermochem | 3273 | 0.0141 |
| 66 | `OHC.PY sensi multiplier[HEFA]_inv` | inverse | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 4561 | 0.2524 |
| 67 | `CHC.initial indices of Commercial Maturity[Brownfield]:CHC.progress ratios commercial[Brownfield]` | interaction | `CHC.initial indices of Commercial Maturity[Brownfield]` | unitless | 0.0 | 0.7 | Cellulosic Thermochem (Brownfield) | 2722 | 0.0182 |
| 68 | `WW.PY sensi multiplier[ManureToHTL]:WW.FCI  sensi multiplier[ManureToHTL]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 1168 | 0.0120 |
| 69 | `WW.progress ratios commercial[SludgeToHTL]:WW.initial indices of Commercial Maturity[SludgeToHTL]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 1990 | 0.0051 |
| 70 | `WW.progress ratios commercial[SludgeToHTL]:AHC.Mature Industry Rate of Return as PCT[HTL]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 908 | 0.0113 |
| 71 | `CHC.PY sensi multiplier[Thermochem]:WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 4980 | 0.0172 |
| 72 | `CHC.Mature Industry Rate of Return as PCT[Thermochem]:WW.progress ratios commercial[SludgeToHTL]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem | 2219 | 0.0162 |
| 73 | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]:WW.ORNOOC sensi multiplier[ManureToHTL]` | interaction | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | %/yr | 5.0 | 15.0 | Manure HTL | 1521 | 0.0047 |
| 74 | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]:WW.FCI  sensi multiplier[ManureToHTL]` | interaction | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | %/yr | 5.0 | 15.0 | Manure HTL | 856 | 0.0110 |
| 75 | `CHC.Mature Industry Rate of Return as PCT[Biochem]:CHC.initial indices of Commercial Maturity[Biochem]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Biochem]` | %/yr | 5.0 | 15.0 | Cellulosic Biochem | 556 | 0.0051 |
| 76 | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]_sq` | quadratic | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | %/yr | 5.0 | 15.0 | Manure HTL | 986 | 0.0036 |
| 77 | `WW.Mature Industry Rate of Return as PCT[SludgeToHTL]:WW.ORNOOC sensi multiplier[SludgeToHTL]` | interaction | `WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | %/yr | 5.0 | 15.0 | Sewage Sludge HTL | 2861 | 0.0103 |
| 78 | `WW.PY sensi multiplier[SludgeToHTL]:AHC.initial indices of Commercial Maturity[HTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 2271 | 0.0041 |
| 79 | `OHC.PY sensi multiplier[HEFA]:OHC.progress ratios commercial[HEFA]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2570 | 0.1323 |
| 80 | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]:WW.progress ratios commercial[ManureToHTL]` | interaction | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | %/yr | 5.0 | 15.0 | Manure HTL | 850 | 0.0064 |
| 81 | `CHC.PY sensi multiplier[Biochem]` | identity | `CHC.PY sensi multiplier[Biochem]` | unitless | 0.75 | 1.25 | Cellulosic Biochem | 846 | 0.0335 |
| 82 | `WW.ORNOOC sensi multiplier[ManureToHTL]:WW.progress ratios commercial[ManureToHTL]` | interaction | `WW.ORNOOC sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 613 | 0.0152 |
| 83 | `CHC.progress ratios commercial[Thermochem]_sq` | quadratic | `CHC.progress ratios commercial[Thermochem]` | 1/doubling | 0.65 | 0.85 | Cellulosic Thermochem | 2512 | 0.0406 |
| 84 | `OHC.Retirement Frac[TransEster]_sq` | quadratic | `OHC.Retirement Frac[TransEster]` | unitless | 0.0 | 0.15 | Oilcrop Transesterification | 3196 | 0.0078 |
| 85 | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]:WW.initial indices of Commercial Maturity[ManureToHTL]` | interaction | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | %/yr | 5.0 | 15.0 | Manure HTL | 1012 | 0.0028 |
| 86 | `CHC.initial indices of Commercial Maturity[Brownfield]:CHC.PY sensi multiplier[Thermochem]` | interaction | `CHC.initial indices of Commercial Maturity[Brownfield]` | unitless | 0.0 | 0.7 | Cellulosic Thermochem (Brownfield) | 3546 | 0.0109 |
| 87 | `AHC.PY sensi multiplier[HTL]:WW.progress ratios commercial[SludgeToHTL]` | interaction | `AHC.PY sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 758 | 0.0256 |
| 88 | `WW.PY sensi multiplier[SludgeToHTL]:CHC.Mature Industry Rate of Return as PCT[Thermochem]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 4570 | 0.0145 |
| 89 | `OHC.PY sensi multiplier[HEFABrownfield]` | identity | `OHC.PY sensi multiplier[HEFABrownfield]` | unitless | 0.75 | 1.25 | Oilcrop HEFA (Brownfield) | 2844 | 0.6184 |
| 90 | `OHC.Retirement Frac[TransEster]:OHC.PY sensi multiplier[HEFABrownfield]` | interaction | `OHC.Retirement Frac[TransEster]` | unitless | 0.0 | 0.15 | Oilcrop Transesterification | 2161 | 0.0206 |
| 91 | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | identity | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | %/yr | 5.0 | 15.0 | Manure HTL | 850 | 0.0260 |
| 92 | `SE.Max Jet Investment Hit Rate:SE.PY sensi multiplier[Jet]` | interaction | `SE.Max Jet Investment Hit Rate` | 1/year | 0.05 | 0.3 | Starch Ethanol-to-Jet | 3191 | 0.0191 |
| 93 | `CHC.PY sensi multiplier[Brownfield]:CHC.FCI  sensi multiplier[Brownfield]` | interaction | `CHC.PY sensi multiplier[Brownfield]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem (Brownfield) | 1746 | 0.0242 |
| 94 | `CHC.PY sensi multiplier[Biochem]:CHC.FCI  sensi multiplier[Biochem]` | interaction | `CHC.PY sensi multiplier[Biochem]` | unitless | 0.75 | 1.25 | Cellulosic Biochem | 842 | 0.0139 |
| 95 | `CHC.initial indices of Commercial Maturity[Biochem]:CHC.PY sensi multiplier[Biochem]` | interaction | `CHC.initial indices of Commercial Maturity[Biochem]` | unitless | 0.0 | 0.2 | Cellulosic Biochem | 744 | 0.0036 |
| 96 | `CHC.Mature Industry Rate of Return as PCT[Brownfield]:CHC.progress ratios commercial[Brownfield]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Brownfield]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem (Brownfield) | 1849 | 0.0141 |
| 97 | `WW.ORNOOC sensi multiplier[SludgeToHTL]` | identity | `WW.ORNOOC sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 2705 | 0.0979 |
| 98 | `CHC.PY sensi multiplier[Thermochem]:CHC.Debt Interest Rate as pct[Thermochem]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 3127 | 0.0090 |
| 99 | `CHC.initial indices of Commercial Maturity[Biochem]:CHC.FCI  sensi multiplier[Biochem]` | interaction | `CHC.initial indices of Commercial Maturity[Biochem]` | unitless | 0.0 | 0.2 | Cellulosic Biochem | 1136 | 0.0037 |
| 100 | `SE.Mature Industry Rate of Return as PCT[Jet]` | identity | `SE.Mature Industry Rate of Return as PCT[Jet]` | %/yr | 5.0 | 15.0 | Starch Ethanol-to-Jet | 848 | 0.0099 |
| 101 | `CHC.Mature Industry Rate of Return as PCT[Thermochem]:WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem | 4682 | 0.0049 |
| 102 | `CHC.PY sensi multiplier[Thermochem]:CHC.Mature Industry Rate of Return as PCT[Brownfield]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 2453 | 0.0110 |
| 103 | `WW.PY sensi multiplier[SludgeToHTL]:CHC.FCI  sensi multiplier[Thermochem]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 3032 | 0.0142 |
| 104 | `WW.PY sensi multiplier[SludgeToHTL]:WW.PY sensi multiplier[ManureToHTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 803 | 0.0048 |
| 105 | `SE.ORNOOC sensi multiplier[Jet]` | identity | `SE.ORNOOC sensi multiplier[Jet]` | unitless | 0.75 | 1.25 | Starch Ethanol-to-Jet | 1158 | 0.0210 |
| 106 | `CHC.FCI  sensi multiplier[Biochem]` | identity | `CHC.FCI  sensi multiplier[Biochem]` | unitless | 0.75 | 1.25 | Cellulosic Biochem | 840 | 0.0146 |
| 107 | `CHC.PY sensi multiplier[Biochem]_inv` | inverse | `CHC.PY sensi multiplier[Biochem]` | unitless | 0.75 | 1.25 | Cellulosic Biochem | 686 | 0.0120 |
| 108 | `SE.PY sensi multiplier[Jet]:SE.Mature Industry Rate of Return as PCT[Jet]` | interaction | `SE.PY sensi multiplier[Jet]` | unitless | 0.75 | 1.25 | Starch Ethanol-to-Jet | 1236 | 0.0063 |
| 109 | `SE.Mature Industry Rate of Return as PCT[Jet]:SE.FCI  sensi multiplier[Jet]` | interaction | `SE.Mature Industry Rate of Return as PCT[Jet]` | %/yr | 5.0 | 15.0 | Starch Ethanol-to-Jet | 1108 | 0.0055 |
| 110 | `WW.PY sensi multiplier[SludgeToHTL]:WW.initial indices of Commercial Maturity[SludgeToHTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 1639 | 0.0022 |
| 111 | `SE.PY sensi multiplier[Jet]_inv` | inverse | `SE.PY sensi multiplier[Jet]` | unitless | 0.75 | 1.25 | Starch Ethanol-to-Jet | 1470 | 0.0198 |
| 112 | `WW.PY sensi multiplier[ManureToHTL]:WW.Policy Duration[Price,ManureToHTL]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 2091 | 0.0050 |
| 113 | `WW.PY sensi multiplier[SludgeToHTL]:CHC.initial indices of Commercial Maturity[Thermochem]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 3589 | 0.0038 |
| 114 | `CHC.PY sensi multiplier[Thermochem]:CHC.ORNOOC sensi multiplier[Thermochem]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 3001 | 0.0123 |
| 115 | `OHC.PY sensi multiplier[HEFA]:OHC.Mature Industry Rate of Return as PCT[HEFA]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2671 | 0.0277 |
| 116 | `AHC.Mature Industry Rate of Return as PCT[HTL]:AHC.FCI  sensi multiplier[HTL]` | interaction | `AHC.Mature Industry Rate of Return as PCT[HTL]` | %/yr | 5.0 | 15.0 | Algal HTL | 674 | 0.0048 |
| 117 | `OHC.Retirement Frac[TransEster]:OHC.PY sensi multiplier[HEFA]` | interaction | `OHC.Retirement Frac[TransEster]` | unitless | 0.0 | 0.15 | Oilcrop Transesterification | 2179 | 0.0140 |
| 118 | `CHC.PY sensi multiplier[Brownfield]_inv` | inverse | `CHC.PY sensi multiplier[Brownfield]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem (Brownfield) | 1622 | 0.0164 |
| 119 | `CHC.initial indices of Commercial Maturity[Brownfield]:CHC.FCI  sensi multiplier[Thermochem]` | interaction | `CHC.initial indices of Commercial Maturity[Brownfield]` | unitless | 0.0 | 0.7 | Cellulosic Thermochem (Brownfield) | 2390 | 0.0035 |
| 120 | `WW.progress ratios commercial[SludgeToHTL]:AHC.FCI  sensi multiplier[HTL]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 1726 | 0.0286 |
| 121 | `CHC.initial indices of Commercial Maturity[Brownfield]:CHC.Mature Industry Rate of Return as PCT[Thermochem]` | interaction | `CHC.initial indices of Commercial Maturity[Brownfield]` | unitless | 0.0 | 0.7 | Cellulosic Thermochem (Brownfield) | 2748 | 0.0029 |
| 122 | `WW.FCI  sensi multiplier[ManureToHTL]` | identity | `WW.FCI  sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 909 | 0.0218 |
| 123 | `OHC.PY sensi multiplier[HEFA]:OHC.ORNOOC sensi multiplier[HEFA]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2498 | 0.0424 |
| 124 | `AHC.PY sensi multiplier[HTL]:AHC.Mature Industry Rate of Return as PCT[HTL]` | interaction | `AHC.PY sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 430 | 0.0054 |
| 125 | `OHC.PY sensi multiplier[HEFA]:CHC.PY sensi multiplier[Thermochem]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 3276 | 0.0523 |
| 126 | `AHC.Mature Industry Rate of Return as PCT[HTL]` | identity | `AHC.Mature Industry Rate of Return as PCT[HTL]` | %/yr | 5.0 | 15.0 | Algal HTL | 829 | 0.0119 |
| 127 | `CHC.FCI  sensi multiplier[Brownfield]` | identity | `CHC.FCI  sensi multiplier[Brownfield]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem (Brownfield) | 706 | 0.0372 |
| 128 | `CHC.PY sensi multiplier[Thermochem]:WW.PY sensi multiplier[ManureToHTL]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 1050 | 0.0046 |
| 129 | `OHC.PY sensi multiplier[HEFA]:OHC.FCI  sensi multiplier[HEFA]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2891 | 0.0373 |
| 130 | `WW.PY sensi multiplier[SludgeToHTL]:WW.ORNOOC sensi multiplier[ManureToHTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 1149 | 0.0033 |
| 131 | `SE.PY sensi multiplier[Jet]:SE.ORNOOC sensi multiplier[Jet]` | interaction | `SE.PY sensi multiplier[Jet]` | unitless | 0.75 | 1.25 | Starch Ethanol-to-Jet | 1378 | 0.0107 |
| 132 | `CHC.PY sensi multiplier[Brownfield]:CHC.progress ratios commercial[Brownfield]` | interaction | `CHC.PY sensi multiplier[Brownfield]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem (Brownfield) | 1602 | 0.0356 |
| 133 | `OHC.PY sensi multiplier[HEFABrownfield]:OHC.progress ratios commercial[HEFA]` | interaction | `OHC.PY sensi multiplier[HEFABrownfield]` | unitless | 0.75 | 1.25 | Oilcrop HEFA (Brownfield) | 2535 | 0.0597 |
| 134 | `OHC.PY sensi multiplier[HEFA]:CHC.Mature Industry Rate of Return as PCT[Thermochem]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 3432 | 0.0196 |
| 135 | `AHC.initial indices of Commercial Maturity[HTL]:WW.initial indices of Commercial Maturity[SludgeToHTL]` | interaction | `AHC.initial indices of Commercial Maturity[HTL]` | unitless | 0.0 | 0.2 | Algal HTL | 4682 | 0.0027 |
| 136 | `AHC.PY sensi multiplier[HTL]:AHC.FCI  sensi multiplier[HTL]` | interaction | `AHC.PY sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 428 | 0.0066 |
| 137 | `AHC.Mature Industry Rate of Return as PCT[HEFA]` | identity | `AHC.Mature Industry Rate of Return as PCT[HEFA]` | %/yr | 5.0 | 15.0 | Algal HEFA | 1086 | 0.0031 |
| 138 | `CHC.Debt Interest Rate as pct[Thermochem]` | identity | `CHC.Debt Interest Rate as pct[Thermochem]` | %/yr | 5.0 | 12.0 | Cellulosic Thermochem | 1279 | 0.0052 |
| 139 | `WW.PY sensi multiplier[SludgeToHTL]:AHC.Mature Industry Rate of Return as PCT[HTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 715 | 0.0026 |
| 140 | `SE.Mature Industry Rate of Return as PCT[Jet]:SE.initial indices of Commercial Maturity[Jet]` | interaction | `SE.Mature Industry Rate of Return as PCT[Jet]` | %/yr | 5.0 | 15.0 | Starch Ethanol-to-Jet | 1160 | 0.0021 |
| 141 | `CHC.PY sensi multiplier[Thermochem]:CHC.PY sensi multiplier[Brownfield]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 1972 | 0.0184 |
| 142 | `CHC.FCI  sensi multiplier[Thermochem]:CHC.Debt Interest Rate as pct[Thermochem]` | interaction | `CHC.FCI  sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 2355 | 0.0039 |
| 143 | `SE.Max Jet Investment Hit Rate:CHC.PY sensi multiplier[Thermochem]` | interaction | `SE.Max Jet Investment Hit Rate` | 1/year | 0.05 | 0.3 | Starch Ethanol-to-Jet | 3934 | 0.0144 |
| 144 | `CHC.ORNOOC sensi multiplier[Thermochem]` | identity | `CHC.ORNOOC sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 2000 | 0.0138 |
| 145 | `OHC.progress ratios commercial[HEFA]:OHC.progress ratios commercial[HEFABrownfield]` | interaction | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 2684 | 0.1134 |
| 146 | `CHC.FCI  sensi multiplier[Brownfield]:CHC.progress ratios commercial[Brownfield]` | interaction | `CHC.FCI  sensi multiplier[Brownfield]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem (Brownfield) | 1360 | 0.0223 |
| 147 | `OHC.PY sensi multiplier[HEFA]:OHC.Policy Duration[Price,HEFA]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 1141 | 0.0123 |
| 148 | `CHC.Mature Industry Rate of Return as PCT[Thermochem]:CHC.Mature Industry Rate of Return as PCT[Brownfield]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem | 2294 | 0.0028 |
| 149 | `OHC.PY sensi multiplier[HEFA]:SE.Max Jet Investment Hit Rate` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2228 | 0.0146 |
| 150 | `CHC.Mature Industry Rate of Return as PCT[Biochem]_inv` | inverse | `CHC.Mature Industry Rate of Return as PCT[Biochem]` | %/yr | 5.0 | 15.0 | Cellulosic Biochem | 607 | 0.0037 |
| 151 | `CHC.FCI  sensi multiplier[Thermochem]:CHC.ORNOOC sensi multiplier[Thermochem]` | interaction | `CHC.FCI  sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 2042 | 0.0063 |
| 152 | `AHC.initial indices of Commercial Maturity[HTL]_sqrt` | sqrt | `AHC.initial indices of Commercial Maturity[HTL]` | unitless | 0.0 | 0.2 | Algal HTL | 2080 | 0.0035 |
| 153 | `AHC.PY sensi multiplier[HTL]:WW.PY sensi multiplier[SludgeToHTL]` | interaction | `AHC.PY sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 2954 | 0.0099 |
| 154 | `OHC.Mature Industry Rate of Return as PCT[HEFA]` | identity | `OHC.Mature Industry Rate of Return as PCT[HEFA]` | %/yr | 5.0 | 15.0 | Oilcrop HEFA | 1663 | 0.0317 |
| 155 | `OHC.PY sensi multiplier[HEFA]:WW.PY sensi multiplier[SludgeToHTL]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 3138 | 0.0238 |
| 156 | `AHC.FCI  sensi multiplier[HEFA]` | identity | `AHC.FCI  sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Algal HEFA | 306 | 0.0111 |
| 157 | `OHC.PY sensi multiplier[HEFA]:CHC.FCI  sensi multiplier[Thermochem]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2404 | 0.0206 |
| 158 | `SE.Max Jet Investment Hit Rate:WW.progress ratios commercial[SludgeToHTL]` | interaction | `SE.Max Jet Investment Hit Rate` | 1/year | 0.05 | 0.3 | Starch Ethanol-to-Jet | 1904 | 0.0186 |
| 159 | `AHC.PY sensi multiplier[HTL]:WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | interaction | `AHC.PY sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 1724 | 0.0028 |
| 160 | `OHC.progress ratios commercial[HEFA]` | identity | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 1518 | 0.4397 |
| 161 | `OHC.PY sensi multiplier[HEFA]:CHC.initial indices of Commercial Maturity[Thermochem]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2982 | 0.0055 |
| 162 | `SE.PY sensi multiplier[Jet]` | identity | `SE.PY sensi multiplier[Jet]` | unitless | 0.75 | 1.25 | Starch Ethanol-to-Jet | 1249 | 0.0363 |
| 163 | `OHC.Retirement Frac[TransEster]:OHC.progress ratios commercial[HEFA]` | interaction | `OHC.Retirement Frac[TransEster]` | unitless | 0.0 | 0.15 | Oilcrop Transesterification | 1228 | 0.0048 |
| 164 | `WW.PY sensi multiplier[ManureToHTL]:WW.ORNOOC sensi multiplier[SludgeToHTL]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 440 | 0.0021 |
| 165 | `OHC.PY sensi multiplier[HEFABrownfield]:SE.Max Jet Investment Hit Rate` | interaction | `OHC.PY sensi multiplier[HEFABrownfield]` | unitless | 0.75 | 1.25 | Oilcrop HEFA (Brownfield) | 2314 | 0.0123 |
| 166 | `OHC.PY sensi multiplier[HEFA]:WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2557 | 0.0103 |
| 167 | `OHC.progress ratios commercial[HEFABrownfield]` | identity | `OHC.progress ratios commercial[HEFABrownfield]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA (Brownfield) | 2215 | 0.1119 |
| 168 | `OHC.Policy Duration[Price,HEFA]` | identity | `OHC.Policy Duration[Price,HEFA]` | year | 18.0 | 36.0 | Oilcrop HEFA | 864 | 0.0138 |
| 169 | `WW.PY sensi multiplier[SludgeToHTL]:AHC.FCI  sensi multiplier[HTL]` | interaction | `WW.PY sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 382 | 0.0037 |
| 170 | `CHC.PY sensi multiplier[Thermochem]:OHC.progress ratios commercial[HEFA]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 1247 | 0.0168 |
| 171 | `OHC.ORNOOC sensi multiplier[HEFA]` | identity | `OHC.ORNOOC sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2213 | 0.0453 |
| 172 | `WW.Policy Duration[Price,ManureToHTL]` | identity | `WW.Policy Duration[Price,ManureToHTL]` | year | 18.0 | 36.0 | Manure HTL | 429 | 0.0055 |
| 173 | `OHC.FCI  sensi multiplier[HEFA]` | identity | `OHC.FCI  sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 2876 | 0.0351 |
| 174 | `CHC.initial indices of Commercial Maturity[Biochem]` | identity | `CHC.initial indices of Commercial Maturity[Biochem]` | unitless | 0.0 | 0.2 | Cellulosic Biochem | 511 | 0.0031 |
| 175 | `AHC.ORNOOC sensi multiplier[HTL]` | identity | `AHC.ORNOOC sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 520 | 0.0218 |
| 176 | `CHC.FCI  sensi multiplier[Thermochem]:CHC.progress ratios commercial[Brownfield]` | interaction | `CHC.FCI  sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 692 | 0.0110 |
| 177 | `SE.Max Jet Investment Hit Rate:SE.progress ratios commercial[Jet]` | interaction | `SE.Max Jet Investment Hit Rate` | 1/year | 0.05 | 0.3 | Starch Ethanol-to-Jet | 1531 | 0.0084 |
| 178 | `OHC.PY sensi multiplier[HEFABrownfield]:CHC.progress ratios commercial[Thermochem]` | interaction | `OHC.PY sensi multiplier[HEFABrownfield]` | unitless | 0.75 | 1.25 | Oilcrop HEFA (Brownfield) | 2611 | 0.0295 |
| 179 | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]:OHC.progress ratios commercial[HEFABrownfield]` | interaction | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | %/yr | 5.0 | 15.0 | Manure HTL | 908 | 0.0040 |
| 180 | `OHC.PY sensi multiplier[HEFA]:WW.ORNOOC sensi multiplier[SludgeToHTL]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 1536 | 0.0076 |
| 181 | `OHC.PY sensi multiplier[HEFABrownfield]:OHC.Policy Duration[Price,HEFABrownfield]` | interaction | `OHC.PY sensi multiplier[HEFABrownfield]` | unitless | 0.75 | 1.25 | Oilcrop HEFA (Brownfield) | 1089 | 0.0086 |
| 182 | `WW.initial indices of Commercial Maturity[ManureToHTL]` | identity | `WW.initial indices of Commercial Maturity[ManureToHTL]` | Unitless | 0.0 | 0.2 | Manure HTL | 949 | 0.0025 |
| 183 | `SE.progress ratios commercial[Jet]` | identity | `SE.progress ratios commercial[Jet]` | 1/doubling | 0.65 | 0.85 | Starch Ethanol-to-Jet | 892 | 0.0141 |
| 184 | `AHC.FCI  sensi multiplier[HTL]` | identity | `AHC.FCI  sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 2736 | 0.0417 |
| 185 | `WW.PY sensi multiplier[ManureToHTL]:SE.progress ratios commercial[Jet]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 278 | 0.0063 |
| 186 | `OHC.ORNOOC sensi multiplier[HEFABrownfield]` | identity | `OHC.ORNOOC sensi multiplier[HEFABrownfield]` | unitless | 0.75 | 1.25 | Oilcrop HEFA (Brownfield) | 654 | 0.0157 |
| 187 | `OHC.PY sensi multiplier[HEFA]:OHC.Debt Interest Rate as pct[HEFA]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 1697 | 0.0060 |
| 188 | `WW.PY sensi multiplier[ManureToHTL]:OHC.ORNOOC sensi multiplier[HEFA]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 1824 | 0.0058 |
| 189 | `OHC.progress ratios commercial[HEFA]:WW.Debt Interest Rate as pct[SludgeToHTL]` | interaction | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 349 | 0.0045 |
| 190 | `OHC.Policy Duration[Price,TransEster]` | identity | `OHC.Policy Duration[Price,TransEster]` | year | 18.0 | 36.0 | Oilcrop Transesterification | 471 | 0.0088 |
| 191 | `OHC.Mature Industry Rate of Return as PCT[HEFA]:OHC.progress ratios commercial[HEFA]` | interaction | `OHC.Mature Industry Rate of Return as PCT[HEFA]` | %/yr | 5.0 | 15.0 | Oilcrop HEFA | 1130 | 0.0062 |
| 192 | `OHC.progress ratios commercial[HEFA]:OHC.ORNOOC sensi multiplier[HEFABrownfield]` | interaction | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 274 | 0.0093 |
| 193 | `CHC.progress ratios commercial[Thermochem]:SE.Policy Duration[Price,Jet]` | interaction | `CHC.progress ratios commercial[Thermochem]` | 1/doubling | 0.65 | 0.85 | Cellulosic Thermochem | 537 | 0.0048 |
| 194 | `CHC.progress ratios commercial[Brownfield]` | identity | `CHC.progress ratios commercial[Brownfield]` | 1/doubling | 0.65 | 0.85 | Cellulosic Thermochem (Brownfield) | 1018 | 0.0927 |
| 195 | `WW.ORNOOC sensi multiplier[ManureToHTL]` | identity | `WW.ORNOOC sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 774 | 0.0236 |
| 196 | `AHC.Debt Interest Rate as pct[HTL]` | identity | `AHC.Debt Interest Rate as pct[HTL]` | %/yr | 5.0 | 12.0 | Algal HTL | 438 | 0.0031 |
| 197 | `CHC.PY sensi multiplier[Thermochem]:CHC.Policy Duration[Price,Thermochem]` | interaction | `CHC.PY sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 2176 | 0.0065 |
| 198 | `WW.PY sensi multiplier[ManureToHTL]:CHC.progress ratios commercial[Thermochem]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 2073 | 0.0196 |
| 199 | `WW.PY sensi multiplier[ManureToHTL]:CHC.PY sensi multiplier[Brownfield]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 568 | 0.0021 |
| 200 | `OHC.progress ratios commercial[HEFA]:OHC.Policy Duration[Price,TransEster]` | interaction | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 252 | 0.0054 |
| 201 | `SE.PY sensi multiplier[Jet]:WW.progress ratios commercial[ManureToHTL]` | interaction | `SE.PY sensi multiplier[Jet]` | unitless | 0.75 | 1.25 | Starch Ethanol-to-Jet | 312 | 0.0073 |
| 202 | `OHC.PY sensi multiplier[HEFA]:WW.PY sensi multiplier[ManureToHTL]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 1434 | 0.0041 |
| 203 | `WW.PY sensi multiplier[ManureToHTL]:WW.Policy Duration[Price,SludgeToHTL]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 934 | 0.0032 |
| 204 | `WW.progress ratios commercial[SludgeToHTL]:OHC.progress ratios commercial[HEFA]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 1253 | 0.1008 |
| 205 | `OHC.Mature Industry Rate of Return as PCT[HEFABrownfield]` | identity | `OHC.Mature Industry Rate of Return as PCT[HEFABrownfield]` | %/yr | 5.0 | 15.0 | Oilcrop HEFA (Brownfield) | 2271 | 0.0047 |
| 206 | `OHC.PY sensi multiplier[HEFA]:SE.Policy Duration[Price,Jet]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 579 | 0.0023 |
| 207 | `OHC.progress ratios commercial[HEFA]:OHC.Policy Duration[Price,HEFABrownfield]` | interaction | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 507 | 0.0064 |
| 208 | `CHC.FCI  sensi multiplier[Brownfield]:OHC.progress ratios commercial[HEFA]` | interaction | `CHC.FCI  sensi multiplier[Brownfield]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem (Brownfield) | 147 | 0.0092 |
| 209 | `CHC.Debt Interest Rate as pct[Biochem]` | identity | `CHC.Debt Interest Rate as pct[Biochem]` | %/yr | 5.0 | 12.0 | Cellulosic Biochem | 411 | 0.0045 |
| 210 | `WW.ORNOOC sensi multiplier[SludgeToHTL]:AHC.ORNOOC sensi multiplier[HTL]` | interaction | `WW.ORNOOC sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 1270 | 0.0050 |
| 211 | `CHC.PY sensi multiplier[Brownfield]:OHC.progress ratios commercial[HEFA]` | interaction | `CHC.PY sensi multiplier[Brownfield]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem (Brownfield) | 232 | 0.0109 |
| 212 | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]:OHC.progress ratios commercial[HEFA]` | interaction | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | %/yr | 5.0 | 15.0 | Manure HTL | 213 | 0.0024 |
| 213 | `WW.progress ratios commercial[ManureToHTL]` | identity | `WW.progress ratios commercial[ManureToHTL]` | 1/doubling | 0.65 | 0.85 | Manure HTL | 246 | 0.0647 |
| 214 | `OHC.Policy Duration[Price,HEFABrownfield]` | identity | `OHC.Policy Duration[Price,HEFABrownfield]` | year | 18.0 | 36.0 | Oilcrop HEFA (Brownfield) | 694 | 0.0204 |
| 215 | `WW.ORNOOC sensi multiplier[SludgeToHTL]:WW.FCI  sensi multiplier[ManureToHTL]` | interaction | `WW.ORNOOC sensi multiplier[SludgeToHTL]` | unitless | 0.75 | 1.25 | Sewage Sludge HTL | 345 | 0.0021 |
| 216 | `WW.Policy Duration[Price,SludgeToHTL]` | identity | `WW.Policy Duration[Price,SludgeToHTL]` | year | 18.0 | 36.0 | Sewage Sludge HTL | 766 | 0.0049 |
| 217 | `SE.ORNOOC sensi multiplier[Jet]:CHC.FCI  sensi multiplier[Brownfield]` | interaction | `SE.ORNOOC sensi multiplier[Jet]` | unitless | 0.75 | 1.25 | Starch Ethanol-to-Jet | 317 | 0.0025 |
| 218 | `WW.Debt Interest Rate as pct[SludgeToHTL]` | identity | `WW.Debt Interest Rate as pct[SludgeToHTL]` | %/yr | 5.0 | 12.0 | Sewage Sludge HTL | 786 | 0.0071 |
| 219 | `CHC.Mature Industry Rate of Return as PCT[Brownfield]:CHC.progress ratios commercial[Thermochem]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Brownfield]` | %/yr | 5.0 | 15.0 | Cellulosic Thermochem (Brownfield) | 990 | 0.0049 |
| 220 | `SE.Policy Duration[Price,Jet]` | identity | `SE.Policy Duration[Price,Jet]` | year | 18.0 | 36.0 | Starch Ethanol-to-Jet | 692 | 0.0090 |
| 221 | `WW.PY sensi multiplier[ManureToHTL]:AHC.FCI  sensi multiplier[HEFA]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 218 | 0.0022 |
| 222 | `OHC.ORNOOC sensi multiplier[HEFA]:AHC.ORNOOC sensi multiplier[HTL]` | interaction | `OHC.ORNOOC sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 675 | 0.0040 |
| 223 | `OHC.progress ratios commercial[HEFA]:CHC.Policy Duration[Price,Thermochem]` | interaction | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 197 | 0.0052 |
| 224 | `AHC.PY sensi multiplier[HTL]:OHC.Policy Duration[Price,HEFABrownfield]` | interaction | `AHC.PY sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 1263 | 0.0031 |
| 225 | `CHC.FCI  sensi multiplier[Brownfield]:OHC.Policy Duration[Price,HEFA]` | interaction | `CHC.FCI  sensi multiplier[Brownfield]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem (Brownfield) | 1894 | 0.0037 |
| 226 | `AHC.ORNOOC sensi multiplier[HTL]:OHC.Policy Duration[Price,HEFABrownfield]` | interaction | `AHC.ORNOOC sensi multiplier[HTL]` | unitless | 0.75 | 1.25 | Algal HTL | 798 | 0.0020 |
| 227 | `OHC.progress ratios commercial[HEFA]:CHC.progress ratios commercial[Brownfield]` | interaction | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 525 | 0.0552 |
| 228 | `WW.PY sensi multiplier[ManureToHTL]:OHC.progress ratios commercial[HEFA]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 261 | 0.0100 |
| 229 | `WW.PY sensi multiplier[ManureToHTL]:CHC.FCI  sensi multiplier[Thermochem]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 362 | 0.0039 |
| 230 | `WW.FCI  sensi multiplier[ManureToHTL]:OHC.progress ratios commercial[HEFA]` | interaction | `WW.FCI  sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 218 | 0.0083 |
| 231 | `CHC.FCI  sensi multiplier[Thermochem]:SE.FCI  sensi multiplier[Jet]` | interaction | `CHC.FCI  sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 169 | 0.0022 |
| 232 | `CHC.Policy Duration[Price,Thermochem]` | identity | `CHC.Policy Duration[Price,Thermochem]` | year | 18.0 | 36.0 | Cellulosic Thermochem | 222 | 0.0168 |
| 233 | `OHC.PY sensi multiplier[HEFA]:CHC.PY sensi multiplier[Biochem]` | interaction | `OHC.PY sensi multiplier[HEFA]` | unitless | 0.75 | 1.25 | Oilcrop HEFA | 354 | 0.0027 |
| 234 | `WW.PY sensi multiplier[ManureToHTL]:OHC.ORNOOC sensi multiplier[HEFABrownfield]` | interaction | `WW.PY sensi multiplier[ManureToHTL]` | unitless | 0.75 | 1.25 | Manure HTL | 506 | 0.0046 |
| 235 | `SE.FCI  sensi multiplier[Jet]` | identity | `SE.FCI  sensi multiplier[Jet]` | unitless | 0.75 | 1.25 | Starch Ethanol-to-Jet | 654 | 0.0093 |
| 236 | `CHC.FCI  sensi multiplier[Thermochem]:OHC.progress ratios commercial[HEFABrownfield]` | interaction | `CHC.FCI  sensi multiplier[Thermochem]` | unitless | 0.75 | 1.25 | Cellulosic Thermochem | 79 | 0.0061 |
| 237 | `WW.progress ratios commercial[SludgeToHTL]:SE.Alt Price Sensi multiplier[EtOH]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 722 | 0.0020 |
| 238 | `OHC.progress ratios commercial[HEFA]:AHC.ORNOOC sensi multiplier[HTL]` | interaction | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 505 | 0.0163 |
| 239 | `WW.progress ratios commercial[SludgeToHTL]:CHC.Policy Duration[Price,Thermochem]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 274 | 0.0099 |
| 240 | `OHC.progress ratios commercial[HEFA]:WW.progress ratios commercial[ManureToHTL]` | interaction | `OHC.progress ratios commercial[HEFA]` | 1/doubling | 0.65 | 0.85 | Oilcrop HEFA | 227 | 0.0447 |
| 241 | `OHC.Debt Interest Rate as pct[HEFA]` | identity | `OHC.Debt Interest Rate as pct[HEFA]` | %/yr | 5.0 | 12.0 | Oilcrop HEFA | 1129 | 0.0073 |
| 242 | `SE.Alt Price Sensi multiplier[EtOH]` | identity | `SE.Alt Price Sensi multiplier[EtOH]` | Unitless | 0.25 | 1.0 | Starch Ethanol | 381 | 0.0042 |
| 243 | `CHC.Mature Industry Rate of Return as PCT[Biochem]:OHC.progress ratios commercial[HEFA]` | interaction | `CHC.Mature Industry Rate of Return as PCT[Biochem]` | %/yr | 5.0 | 15.0 | Cellulosic Biochem | 511 | 0.0028 |
| 244 | `WW.progress ratios commercial[SludgeToHTL]:OHC.Mature Industry Rate of Return as PCT[HEFA]` | interaction | `WW.progress ratios commercial[SludgeToHTL]` | 1/doubling | 0.65 | 0.85 | Sewage Sludge HTL | 338 | 0.0025 |

## How to use this table

1. **Reading the coefficient matrix:** column at index `i` of the coefficient matrix corresponds to the feature at row `i` of this table.
2. **Reading a coefficient value:** the raw-scale coefficient maps to the base input in its native units; the standardized coefficient maps to the base input rescaled by its training-partition mean and standard deviation (see `x_standardization.csv`, `y_standardization.csv`).
3. **Interaction columns:** the product `x_a * x_b` is computed AFTER any per-side transformation (e.g. `quadratic_X:Y` = `X^2 * Y`).
4. **Unit conventions:** any unitless input has `units = 'unitless'`. Interactions and transformed inputs inherit the product/composition of their constituent units; the raw coefficient value carries the inverse of those units to recover the output unit.
