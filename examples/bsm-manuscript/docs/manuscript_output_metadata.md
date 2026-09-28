# Manuscript Output Metadata

Comprehensive metadata for every base output variable predicted by the BSM reduced-form model, derived from Appendix B (Table B1) of the FY25Q4 technical report. The 17 base outputs expand into 23,495 scalar time-series outputs after arraying and annualization (2015-2051).

| # | Variable | Description | Unit | Array specification |
| - | -------- | ----------- | ---- | ------------------- |
| 1 | `AHC.MFSPMetric` | Minimum Fuel Selling Price metric for algal fuels | USD/gal | Arrayed by conversion pathway and region |
| 2 | `CHC.MFSPMetric` | Minimum Fuel Selling Price metric for cellulosic fuels | USD/gal | Arrayed by conversion pathway and region |
| 3 | `OHC.MFSPMetric` | Minimum Fuel Selling Price metric for oilcrop and FOG-derived fuels | USD/gal | Arrayed by conversion pathway and region |
| 4 | `OI.AHC output by product` | Annual production of biofuels from algal feedstocks | Gal/yr | Arrayed by conversion pathway, region, and product (i.e., gasoline, diesel, jet) |
| 5 | `OI.CHC output by product` | Annual production of biofuels from cellulosic feedstocks | Gal/yr | Arrayed by conversion pathway, region, and product (i.e., gasoline, diesel, jet) |
| 6 | `OI.OHC output by product` | Annual production of biofuels from oilcrop and FOG feedstocks | Gal/yr | Arrayed by region and product (i.e., gasoline,  jet) |
| 7 | `OI.SJ output by product` | Annual production of jet fuel from starch-ethanol | Gal/yr | Arrayed by conversion pathway, region, and product (i.e., gasoline, diesel, jet) |
| 8 | `OI.WW output by product` | Annual production of biofuels from wet waste feedstocks | Gal/yr | Arrayed by conversion pathway, region, and product (i.e., gasoline, diesel, jet) |
| 9 | `OI.annual EtOH prodn by region` | Annual production of ethanol from starch and cellulosic feedstocks | Gal/yr | Arrayed by region |
| 10 | `OI.annual HCBN prodn by product` | Total production of hydrocarbons from all conversion platforms | Gal/yr | Arrayed by product (i.e., gasoline, diesel, jet) |
| 11 | `OI.annual total HCBN prodn` | Total production of hydrocarbons. Sums production across platforms, products | Gal/yr | Scalar |
| 12 | `OI.composite EtOH CI` | Composite metric for carbon intensity of ethanol from starch and cellulosic feedstocks | g/MJ | Arrayed by region and product (i.e., gasoline, diesel, jet) |
| 13 | `OI.composite EtOH MFSP` | Composite metric for minimum fuel selling price of ethanol from starch and cellulosic feedstocks | USD/gal | Arrayed by region |
| 14 | `OI.composite HCBN CI` | Composite metric for carbon intensity of hydrocarbon fuels from all modeled hydrocarbon fuels | g/MJ | Arrayed by region and product (e.g., gasoline, diesel, jet) |
| 15 | `OI.composite HCBN MFSP` | Composite metric for minimum fuel selling price of hydrocarbons from all conversion platforms | USD/gal | Arrayed by region |
| 16 | `SE.MFSPMetric` | Metric for minimum fuel selling price of starch ethanol and ethanol-to-jet fuel | USD/gal | Arrayed by region, product (i.e., ethanol, jet fuel) |
| 17 | `WW.MFSPMetric` | Minimum Fuel Selling Price metric for wet waste-derived fuels | USD/gal | Arrayed by conversion pathway and region |

## Output expansion convention

Each base output is arrayed across one or more discrete dimensions (conversion pathway, region, product). When the arrayed scalars are evaluated for each of the 37 annual values (2015 – 2051), the total output space is **23,495 scalars**. Of these, **9,954** exhibit non-negligible variance under the LHC input sweep and are scored by the manuscript's reported nRMSE; the remaining outputs are predicted with near-zero coefficients (constant or near-constant series).
