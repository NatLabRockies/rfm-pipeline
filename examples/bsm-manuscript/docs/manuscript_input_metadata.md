# Manuscript Input Metadata

Comprehensive metadata for every first-order input to the BSM reduced-form model, derived from Appendix A (Table A1) of the FY25Q4 technical report.

**Inputs in this catalog:** 160 (2 binary scenario switches + 158 continuous LHC-sampled inputs).

**Source of truth.** Variable names, descriptions, pathway tags, sample ranges, and units come directly from Appendix A Table A1. Module abbreviations come from `manuscript_module_abbreviations.yml`. Inputs are grouped by module prefix in document order.

## Module `AHC` — Algal Hydrocarbons

Conversion pathways producing hydrocarbon fuels from algal feedstocks (HEFA, HTL).

| # | Input name | Description | Pathway | Min | Max | Units |
| - | ---------- | ----------- | ------- | --- | --- | ----- |
| 1 | `AHC.Alt Price Sensi multiplier[HEFA]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Algal HEFA | 0.25 | 2.0 | unitless |
| 2 | `AHC.Alt Price Sensi multiplier[HTL]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Algal HTL | 0.25 | 2.0 | unitless |
| 3 | `AHC.Debt Interest Rate as pct[HEFA]` | Interest rate of loan for FCI of conversion facility | Algal HEFA | 5.0 | 12.0 | %/yr |
| 4 | `AHC.Debt Interest Rate as pct[HTL]` | Interest rate of loan for FCI of conversion facility | Algal HTL | 5.0 | 12.0 | %/yr |
| 5 | `AHC.FCI  sensi multiplier[HEFA]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Algal HEFA | 0.75 | 1.25 | unitless |
| 6 | `AHC.FCI  sensi multiplier[HTL]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Algal HTL | 0.75 | 1.25 | unitless |
| 7 | `AHC.FCI Background Subs[HEFA]` | Sets fraction of FCI to be funded by external incentive | Algal HEFA | 0.0 | 0.5 | unitless |
| 8 | `AHC.FCI Background Subs[HTL]` | Sets fraction of FCI to be funded by external incentive | Algal HTL | 0.0 | 0.5 | unitless |
| 9 | `AHC.FS Background Subs[HEFA]` | Sets size of incentive that reduces delivered cost of feedstock | Algal HEFA | 0.0 | 50.0 | USD/ton |
| 10 | `AHC.FS Background Subs[HTL]` | Sets size of incentive that reduces delivered cost of feedstock | Algal HTL | 0.0 | 50.0 | USD/ton |
| 11 | `AHC.initial indices of Commercial Maturity[HEFA]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Algal HEFA | 0.6 | 0.8 | unitless |
| 12 | `AHC.initial indices of Commercial Maturity[HTL]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Algal HTL | 0.0 | 0.2 | unitless |
| 13 | `AHC.Loan Guarantee Background Subs[HEFA]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Algal HEFA | 0.0 | 0.5 | unitless |
| 14 | `AHC.Loan Guarantee Background Subs[HTL]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Algal HTL | 0.0 | 0.5 | unitless |
| 15 | `AHC.Mature Industry Rate of Return as PCT[HEFA]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Algal HEFA | 5.0 | 15.0 | %/yr |
| 16 | `AHC.Mature Industry Rate of Return as PCT[HTL]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Algal HTL | 5.0 | 15.0 | %/yr |
| 17 | `AHC.ORNOOC sensi multiplier[HEFA]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Algal HEFA | 0.75 | 1.25 | unitless |
| 18 | `AHC.ORNOOC sensi multiplier[HTL]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Algal HTL | 0.75 | 1.25 | unitless |
| 19 | `AHC.Policy Duration[FCI,HEFA]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Algal HEFA | 10.0 | 28.0 | year |
| 20 | `AHC.Policy Duration[FCI,HTL]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Algal HTL | 10.0 | 28.0 | year |
| 21 | `AHC.Policy Duration[Feedstock,HEFA]` | Sets duration of Feedstock incentive beginning with initial simulation year (2015) | Algal HEFA | 10.0 | 28.0 | year |
| 22 | `AHC.Policy Duration[Feedstock,HTL]` | Sets duration of Feedstock incentive beginning with initial simulation year (2015) | Algal HTL | 10.0 | 28.0 | year |
| 23 | `AHC.Policy Duration[Loan,HEFA]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Algal HEFA | 10.0 | 28.0 | year |
| 24 | `AHC.Policy Duration[Loan,HTL]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Algal HTL | 10.0 | 28.0 | year |
| 25 | `AHC.Policy Duration[Price,HEFA]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Algal HEFA | 18.0 | 36.0 | year |
| 26 | `AHC.Policy Duration[Price,HTL]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Algal HTL | 18.0 | 36.0 | year |
| 27 | `AHC.progress ratios commercial[HEFA]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Algal HEFA | 0.65 | 0.85 | unitless |
| 28 | `AHC.progress ratios commercial[HTL]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Algal HTL | 0.65 | 0.85 | 1/doubling |
| 29 | `AHC.PY sensi multiplier[HEFA]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Algal HEFA | 0.75 | 1.25 | unitless |
| 30 | `AHC.PY sensi multiplier[HTL]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Algal HTL | 0.75 | 1.25 | unitless |

## Module `CHC` — Cellulosic Hydrocarbons

Conversion pathways producing hydrocarbon fuels from cellulosic feedstocks (Biochem, Thermochem, Brownfield).

| # | Input name | Description | Pathway | Min | Max | Units |
| - | ---------- | ----------- | ------- | --- | --- | ----- |
| 1 | `CHC.Alt Price Sensi multiplier[Biochem]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Cellulosic Biochem | 0.25 | 2.0 | unitless |
| 2 | `CHC.Alt Price Sensi multiplier[Brownfield]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Cellulosic Thermochem (Brownfield) | 0.0 | 2.0 | unitless |
| 3 | `CHC.Alt Price Sensi multiplier[Thermochem]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Cellulosic Thermochem | 0.25 | 2.0 | unitless |
| 4 | `CHC.Debt Interest Rate as pct[Biochem]` | Interest rate of loan for FCI of conversion facility | Cellulosic Biochem | 5.0 | 12.0 | %/yr |
| 5 | `CHC.Debt Interest Rate as pct[Brownfield]` | Interest rate of loan for FCI of conversion facility | Cellulosic Thermochem (Brownfield) | 5.0 | 12.0 | %/yr |
| 6 | `CHC.Debt Interest Rate as pct[Thermochem]` | Interest rate of loan for FCI of conversion facility | Cellulosic Thermochem | 5.0 | 12.0 | %/yr |
| 7 | `CHC.FCI  sensi multiplier[Biochem]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Cellulosic Biochem | 0.75 | 1.25 | unitless |
| 8 | `CHC.FCI  sensi multiplier[Brownfield]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Cellulosic Thermochem (Brownfield) | 0.75 | 1.25 | unitless |
| 9 | `CHC.FCI  sensi multiplier[Thermochem]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Cellulosic Thermochem | 0.75 | 1.25 | unitless |
| 10 | `CHC.FCI Background Subs[Biochem]` | Sets fraction of FCI to be funded by external incentive | Cellulosic Biochem | 0.0 | 0.5 | unitless |
| 11 | `CHC.FCI Background Subs[Brownfield]` | Sets fraction of FCI to be funded by external incentive | Cellulosic Thermochem (Brownfield) | 0.0 | 0.5 | unitless |
| 12 | `CHC.FCI Background Subs[Thermochem]` | Sets fraction of FCI to be funded by external incentive | Cellulosic Thermochem | 0.0 | 0.5 | unitless |
| 13 | `CHC.FS Background Subs[Biochem]` | Sets size of incentive that reduces delivered cost of feedstock | Cellulosic Biochem | 0.0 | 50.0 | USD/ton |
| 14 | `CHC.FS Background Subs[Brownfield]` | Sets size of incentive that reduces delivered cost of feedstock | Cellulosic Thermochem (Brownfield) | 0.0 | 50.0 | USD/ton |
| 15 | `CHC.FS Background Subs[Thermochem]` | Sets size of incentive that reduces delivered cost of feedstock | Cellulosic Thermochem | 0.0 | 50.0 | USD/ton |
| 16 | `CHC.incremental scale up level[Biochem]` | Sets maximum increment in conversion facility scale relative to nth plant TEA values | Cellulosic Biochem | 0.0 | 2.0 | unitless |
| 17 | `CHC.incremental scale up level[Thermochem]` | Sets maximum increment in conversion facility scale relative to nth plant TEA values | Cellulosic Thermochem | 0.0 | 2.0 | unitless |
| 18 | `CHC.initial indices of Commercial Maturity[Biochem]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Cellulosic Biochem | 0.0 | 0.2 | unitless |
| 19 | `CHC.initial indices of Commercial Maturity[Brownfield]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Cellulosic Thermochem (Brownfield) | 0.0 | 0.7 | unitless |
| 20 | `CHC.initial indices of Commercial Maturity[Thermochem]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Cellulosic Thermochem | 0.1 | 0.7 | unitless |
| 21 | `CHC.Loan Guarantee Background Subs[Biochem]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Cellulosic Biochem | 0.0 | 0.5 | unitless |
| 22 | `CHC.Loan Guarantee Background Subs[Brownfield]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Cellulosic Thermochem (Brownfield) | 0.0 | 0.5 | unitless |
| 23 | `CHC.Loan Guarantee Background Subs[Thermochem]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Cellulosic Thermochem | 0.0 | 0.5 | unitless |
| 24 | `CHC.Mature Industry Rate of Return as PCT[Biochem]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Cellulosic Biochem | 5.0 | 15.0 | %/yr |
| 25 | `CHC.Mature Industry Rate of Return as PCT[Brownfield]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Cellulosic Thermochem (Brownfield) | 5.0 | 15.0 | %/yr |
| 26 | `CHC.Mature Industry Rate of Return as PCT[Thermochem]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Cellulosic Thermochem | 5.0 | 15.0 | %/yr |
| 27 | `CHC.ORNOOC sensi multiplier[Biochem]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Cellulosic Biochem | 0.75 | 1.25 | unitless |
| 28 | `CHC.ORNOOC sensi multiplier[Brownfield]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Cellulosic Thermochem (Brownfield) | 0.75 | 1.25 | unitless |
| 29 | `CHC.ORNOOC sensi multiplier[Thermochem]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Cellulosic Thermochem | 0.75 | 1.25 | unitless |
| 30 | `CHC.Policy Duration[FCI,Biochem]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Cellulosic Biochem | 10.0 | 28.0 | year |
| 31 | `CHC.Policy Duration[FCI,Brownfield]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Cellulosic Thermochem (Brownfield) | 10.0 | 28.0 | year |
| 32 | `CHC.Policy Duration[FCI,Thermochem]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Cellulosic Thermochem | 10.0 | 28.0 | year |
| 33 | `CHC.Policy Duration[Feedstock,Biochem]` | Sets duration of Feedstock incentive beginning with initial simulation year (2015) | Cellulosic Biochem | 10.0 | 28.0 | year |
| 34 | `CHC.Policy Duration[Feedstock,Brownfield]` | Sets duration of Feedstock incentive beginning with initial simulation year (2015) | Cellulosic Thermochem (Brownfield) | 10.0 | 28.0 | year |
| 35 | `CHC.Policy Duration[Feedstock,Thermochem]` | Sets duration of Feedstock incentive beginning with initial simulation year (2015) | Cellulosic Thermochem | 10.0 | 28.0 | year |
| 36 | `CHC.Policy Duration[Loan,Biochem]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Cellulosic Biochem | 10.0 | 28.0 | year |
| 37 | `CHC.Policy Duration[Loan,Brownfield]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Cellulosic Thermochem (Brownfield) | 10.0 | 28.0 | year |
| 38 | `CHC.Policy Duration[Loan,Thermochem]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Cellulosic Thermochem | 10.0 | 28.0 | year |
| 39 | `CHC.Policy Duration[Price,Biochem]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Cellulosic Biochem | 18.0 | 36.0 | year |
| 40 | `CHC.Policy Duration[Price,Brownfield]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Cellulosic Thermochem (Brownfield) | 18.0 | 36.0 | year |
| 41 | `CHC.Policy Duration[Price,Thermochem]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Cellulosic Thermochem | 18.0 | 36.0 | year |
| 42 | `CHC.progress ratios commercial[Biochem]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Cellulosic Biochem | 0.65 | 0.85 | 1/doubling |
| 43 | `CHC.progress ratios commercial[Brownfield]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Cellulosic Thermochem (Brownfield) | 0.65 | 0.85 | 1/doubling |
| 44 | `CHC.progress ratios commercial[Thermochem]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Cellulosic Thermochem | 0.65 | 0.85 | 1/doubling |
| 45 | `CHC.PY sensi multiplier[Biochem]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Cellulosic Biochem | 0.75 | 1.25 | unitless |
| 46 | `CHC.PY sensi multiplier[Brownfield]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Cellulosic Thermochem (Brownfield) | 0.75 | 1.25 | unitless |
| 47 | `CHC.PY sensi multiplier[Thermochem]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Cellulosic Thermochem | 0.75 | 1.25 | unitless |

## Module `OHC` — Oil Hydrocarbons

Conversion pathways producing hydrocarbon fuels from oilcrop and FOG (fats/oils/greases) feedstocks (HEFA, HEFA-Brownfield).

| # | Input name | Description | Pathway | Min | Max | Units |
| - | ---------- | ----------- | ------- | --- | --- | ----- |
| 1 | `OHC.Alt Price Sensi multiplier[HEFA]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Oilcrop HEFA | 0.25 | 2.0 | unitless |
| 2 | `OHC.Alt Price Sensi multiplier[HEFABrownfield]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Oilcrop HEFA (Brownfield) | 0.25 | 2.0 | unitless |
| 3 | `OHC.Alt Price Sensi multiplier[TransEster]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Oilcrop Transesterification | 0.0 | 1.0 | unitless |
| 4 | `OHC.Debt Interest Rate as pct[HEFA]` | Interest rate of loan for FCI of conversion facility | Oilcrop HEFA | 5.0 | 12.0 | %/yr |
| 5 | `OHC.Debt Interest Rate as pct[HEFABrownfield]` | Interest rate of loan for FCI of conversion facility | Oilcrop HEFA (Brownfield) | 5.0 | 12.0 | %/yr |
| 6 | `OHC.FCI  sensi multiplier[HEFA]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Oilcrop HEFA | 0.75 | 1.25 | unitless |
| 7 | `OHC.FCI  sensi multiplier[HEFABrownfield]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Oilcrop HEFA (Brownfield) | 0.75 | 1.25 | unitless |
| 8 | `OHC.FCI Background Subs[HEFA]` | Sets fraction of FCI to be funded by external incentive | Oilcrop HEFA | 0.0 | 0.5 | unitless |
| 9 | `OHC.FCI Background Subs[HEFABrownfield]` | Sets fraction of FCI to be funded by external incentive | Oilcrop HEFA (Brownfield) | 0.0 | 0.5 | unitless |
| 10 | `OHC.FS Background Subs[HEFA]` | Sets size of incentive that reduces delivered cost of feedstock | Oilcrop HEFA | 0.0 | 50.0 | USD/ton |
| 11 | `OHC.FS Background Subs[HEFABrownfield]` | Sets size of incentive that reduces delivered cost of feedstock | Oilcrop HEFA (Brownfield) | 0.0 | 50.0 | USD/ton |
| 12 | `OHC.incremental scale up level[HEFA]` | Sets maximum increment in conversion facility scale relative to nth plant TEA values | Oilcrop HEFA | 0.0 | 1.0 | unitless |
| 13 | `OHC.initial indices of Commercial Maturity[HEFA]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Oilcrop HEFA | 0.6 | 0.8 | unitless |
| 14 | `OHC.initial indices of Commercial Maturity[HEFABrownfield]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Oilcrop HEFA (Brownfield) | 0.6 | 0.8 | unitless |
| 15 | `OHC.Loan Guarantee Background Subs[HEFA]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Oilcrop HEFA | 0.0 | 0.5 | unitless |
| 16 | `OHC.Loan Guarantee Background Subs[HEFABrownfield]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Oilcrop HEFA (Brownfield) | 0.0 | 0.5 | unitless |
| 17 | `OHC.Mature Industry Rate of Return as PCT[HEFA]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Oilcrop HEFA | 5.0 | 15.0 | %/yr |
| 18 | `OHC.Mature Industry Rate of Return as PCT[HEFABrownfield]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Oilcrop HEFA (Brownfield) | 5.0 | 15.0 | %/yr |
| 19 | `OHC.ORNOOC sensi multiplier[HEFA]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Oilcrop HEFA | 0.75 | 1.25 | unitless |
| 20 | `OHC.ORNOOC sensi multiplier[HEFABrownfield]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Oilcrop HEFA (Brownfield) | 0.75 | 1.25 | unitless |
| 21 | `OHC.Policy Duration[FCI,HEFA]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Oilcrop HEFA | 10.0 | 28.0 | year |
| 22 | `OHC.Policy Duration[FCI,HEFABrownfield]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Oilcrop HEFA (Brownfield) | 10.0 | 28.0 | year |
| 23 | `OHC.Policy Duration[Feedstock,HEFA]` | Sets duration of Feedstock incentive beginning with initial simulation year (2015) | Oilcrop HEFA | 10.0 | 28.0 | year |
| 24 | `OHC.Policy Duration[Feedstock,HEFABrownfield]` | Sets duration of Feedstock incentive beginning with initial simulation year (2015) | Oilcrop HEFA (Brownfield) | 10.0 | 28.0 | year |
| 25 | `OHC.Policy Duration[Loan,HEFA]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Oilcrop HEFA | 10.0 | 28.0 | year |
| 26 | `OHC.Policy Duration[Loan,HEFABrownfield]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Oilcrop HEFA (Brownfield) | 10.0 | 28.0 | year |
| 27 | `OHC.Policy Duration[Price,HEFA]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Oilcrop HEFA | 18.0 | 36.0 | year |
| 28 | `OHC.Policy Duration[Price,HEFABrownfield]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Oilcrop HEFA (Brownfield) | 18.0 | 36.0 | year |
| 29 | `OHC.Policy Duration[Price,TransEster]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Oilcrop Transesterification | 18.0 | 36.0 | year |
| 30 | `OHC.progress ratios commercial[HEFA]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Oilcrop HEFA | 0.65 | 0.85 | 1/doubling |
| 31 | `OHC.progress ratios commercial[HEFABrownfield]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Oilcrop HEFA (Brownfield) | 0.65 | 0.85 | 1/doubling |
| 32 | `OHC.PY sensi multiplier[HEFA]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Oilcrop HEFA | 0.75 | 1.25 | unitless |
| 33 | `OHC.PY sensi multiplier[HEFABrownfield]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Oilcrop HEFA (Brownfield) | 0.75 | 1.25 | unitless |
| 34 | `OHC.Retirement Frac[TransEster]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Oilcrop Transesterification | 0.0 | 0.15 | unitless |

## Module `SE` — Starch Ethanol

Conversion pathways producing ethanol and ethanol-to-jet fuel from starch feedstocks.

| # | Input name | Description | Pathway | Min | Max | Units |
| - | ---------- | ----------- | ------- | --- | --- | ----- |
| 1 | `SE.Alt Price Sensi multiplier[EtOH]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Starch Ethanol | 0.25 | 1.0 | Unitless |
| 2 | `SE.Alt Price Sensi multiplier[Jet]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Starch Ethanol-to-Jet | 0.25 | 2.0 | unitless |
| 3 | `SE.Debt Interest Rate as pct[Jet]` | Interest rate of loan for FCI of conversion facility | Starch Ethanol-to-Jet | 5.0 | 12.0 | %/yr |
| 4 | `SE.FCI  sensi multiplier[Jet]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Starch Ethanol-to-Jet | 0.75 | 1.25 | unitless |
| 5 | `SE.FCI Background Subs[Jet]` | Sets fraction of FCI to be funded by external incentive | Starch Ethanol-to-Jet | 0.0 | 0.5 | unitless |
| 6 | `SE.FS Background Subs[Jet]` | Sets size of incentive that reduces delivered cost of feedstock | Starch Ethanol-to-Jet | 0.0 | 50.0 | USD/ton |
| 7 | `SE.initial indices of Commercial Maturity[Jet]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Starch Ethanol-to-Jet | 0.1 | 0.5 | unitless |
| 8 | `SE.Loan Guarantee Background Subs[Jet]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Starch Ethanol-to-Jet | 0.0 | 0.5 | unitless |
| 9 | `SE.Mature Industry Rate of Return as PCT[Jet]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Starch Ethanol-to-Jet | 5.0 | 15.0 | %/yr |
| 10 | `SE.Max Jet Investment Hit Rate` | Sets maximum potential rate of conversion of starch ethanol facilities to production of jet fuel | Starch Ethanol-to-Jet | 0.05 | 0.3 | 1/year |
| 11 | `SE.ORNOOC sensi multiplier[Jet]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Starch Ethanol-to-Jet | 0.75 | 1.25 | unitless |
| 12 | `SE.Policy Duration[FCI,Jet]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Starch Ethanol-to-Jet | 10.0 | 28.0 | year |
| 13 | `SE.Policy Duration[Feedstock,Jet]` | Sets duration of Feedstock incentive beginning with initial simulation year (2015) | Starch Ethanol-to-Jet | 10.0 | 28.0 | year |
| 14 | `SE.Policy Duration[Loan,Jet]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Starch Ethanol-to-Jet | 10.0 | 28.0 | year |
| 15 | `SE.Policy Duration[Price,Jet]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Starch Ethanol-to-Jet | 18.0 | 36.0 | year |
| 16 | `SE.progress ratios commercial[Jet]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Starch Ethanol-to-Jet | 0.65 | 0.85 | 1/doubling |
| 17 | `SE.PY sensi multiplier[Jet]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Starch Ethanol-to-Jet | 0.75 | 1.25 | unitless |

## Module `WW` — Wet Waste Hydrocarbons

Conversion pathways producing hydrocarbon fuels from wet-waste feedstocks (ManureToHTL, SludgeToHTL).

| # | Input name | Description | Pathway | Min | Max | Units |
| - | ---------- | ----------- | ------- | --- | --- | ----- |
| 1 | `WW.Alt Price Sensi multiplier[ManureToHTL]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Manure HTL | 0.25 | 2.0 | unitless |
| 2 | `WW.Alt Price Sensi multiplier[SludgeToHTL]` | Multiplier applied to 2024 value used for unit price incentive for fuels | Sewage Sludge HTL | 0.25 | 2.0 | unitless |
| 3 | `WW.Debt Interest Rate as pct[ManureToHTL]` | Interest rate of loan for FCI of conversion facility | Manure HTL | 5.0 | 12.0 | %/yr |
| 4 | `WW.Debt Interest Rate as pct[SludgeToHTL]` | Interest rate of loan for FCI of conversion facility | Sewage Sludge HTL | 5.0 | 12.0 | %/yr |
| 5 | `WW.FCI  sensi multiplier[ManureToHTL]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Manure HTL | 0.75 | 1.25 | unitless |
| 6 | `WW.FCI  sensi multiplier[SludgeToHTL]` | Multiplier that is applied to fixed capital investment cost for conversion facility | Sewage Sludge HTL | 0.75 | 1.25 | unitless |
| 7 | `WW.FCI Background Subs[ManureToHTL]` | Sets fraction of FCI to be funded by external incentive | Manure HTL | 0.0 | 0.5 | unitless |
| 8 | `WW.FCI Background Subs[SludgeToHTL]` | Sets fraction of FCI to be funded by external incentive | Sewage Sludge HTL | 0.0 | 0.5 | unitless |
| 9 | `WW.FS Background Subs[ManureToHTL]` | Sets size of incentive that reduces delivered cost of feedstock | Manure HTL | 0.0 | 50.0 | USD/ton |
| 10 | `WW.FS Background Subs[SludgeToHTL]` | Sets size of incentive that reduces delivered cost of feedstock | Sewage Sludge HTL | 0.0 | 50.0 | USD/ton |
| 11 | `WW.initial indices of Commercial Maturity[ManureToHTL]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Manure HTL | 0.0 | 0.2 | Unitless |
| 12 | `WW.initial indices of Commercial Maturity[SludgeToHTL]` | Sets initial value for commercial maturity for conversion pathway on 0-1 scale | Sewage Sludge HTL | 0.0 | 0.2 | Unitless |
| 13 | `WW.Loan Guarantee Background Subs[ManureToHTL]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Manure HTL | 0.0 | 0.5 | unitless |
| 14 | `WW.Loan Guarantee Background Subs[SludgeToHTL]` | Sets maximum fraction of FCI that is amenable to loan guarantee | Sewage Sludge HTL | 0.0 | 0.5 | unitless |
| 15 | `WW.Mature Industry Rate of Return as PCT[ManureToHTL]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Manure HTL | 5.0 | 15.0 | %/yr |
| 16 | `WW.Mature Industry Rate of Return as PCT[SludgeToHTL]` | Sets required rate of return on investment for conversion facility. Used in calculation of net present value of prospective investment | Sewage Sludge HTL | 5.0 | 15.0 | %/yr |
| 17 | `WW.ORNOOC sensi multiplier[ManureToHTL]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Manure HTL | 0.75 | 1.25 | unitless |
| 18 | `WW.ORNOOC sensi multiplier[SludgeToHTL]` | Multiplier applied to fixed operating cost, other (non-feedstock) operating costs, power sale credit, and other coproduct sale credit | Sewage Sludge HTL | 0.75 | 1.25 | unitless |
| 19 | `WW.Policy Duration[FCI,ManureToHTL]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Manure HTL | 10.0 | 28.0 | year |
| 20 | `WW.Policy Duration[FCI,SludgeToHTL]` | Sets duration of FCI incentive beginning with initial simulation year (2015) | Sewage Sludge HTL | 10.0 | 28.0 | year |
| 21 | `WW.Policy Duration[Feedstock,ManureToHTL]` | Sets duration of Feedstock cost incentive beginning with initial simulation year (2015) | Manure HTL | 10.0 | 28.0 | year |
| 22 | `WW.Policy Duration[Feedstock,SludgeToHTL]` | Sets duration of Feedstock cost incentive beginning with initial simulation year (2015) | Sewage Sludge HTL | 10.0 | 28.0 | year |
| 23 | `WW.Policy Duration[Loan,ManureToHTL]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Manure HTL | 10.0 | 28.0 | year |
| 24 | `WW.Policy Duration[Loan,SludgeToHTL]` | Sets duration of loan guarantee incentive beginning with initial simulation year (2015) | Sewage Sludge HTL | 10.0 | 28.0 | year |
| 25 | `WW.Policy Duration[Price,ManureToHTL]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Manure HTL | 18.0 | 36.0 | year |
| 26 | `WW.Policy Duration[Price,SludgeToHTL]` | Sets duration of unit price related incentive beginning with initial simulation year (2015) | Sewage Sludge HTL | 18.0 | 36.0 | year |
| 27 | `WW.progress ratios commercial[ManureToHTL]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Manure HTL | 0.65 | 0.85 | 1/doubling |
| 28 | `WW.progress ratios commercial[SludgeToHTL]` | Sets rate at which each doubling of cumulative production impacts commercial maturity of conversion pathway | Sewage Sludge HTL | 0.65 | 0.85 | 1/doubling |
| 29 | `WW.PY sensi multiplier[ManureToHTL]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Manure HTL | 0.75 | 1.25 | unitless |
| 30 | `WW.PY sensi multiplier[SludgeToHTL]` | Multiplier applied to conversion process yield (i.e., gallons fuel per ton feedstock) | Sewage Sludge HTL | 0.75 | 1.25 | unitless |

## Module `FM` — Feedstock Module

Cross-cutting feedstock allocation and policy knobs (binary scenario switches).

| # | Input name | Description | Pathway | Min | Max | Units |
| - | ---------- | ----------- | ------- | --- | --- | ----- |
| 1 | `FM.Use Agnostic FS Conversion` | Binary switch. Agnostic  Herbaceous and Woody feedstocks usable by all processes. Target  Herbaceous to Biochem, Woody to Thermochem | Applies to Cellulosic Hydrocarbons | — | — | Binary flags (0= off; 1 = on)* |

## Module `OI` — Output Indicators

Aggregate output variables (production, MFSP, carbon intensity) reported by the BSM across all conversion platforms.

| # | Input name | Description | Pathway | Min | Max | Units |
| - | ---------- | ----------- | ------- | --- | --- | ----- |
| 1 | `OI.Use AEO Reference Oil` | Switch to use AEO reference vs high projection for oil price | - | — | — | unitless |
