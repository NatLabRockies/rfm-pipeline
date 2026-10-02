# BSM application

The [BSM Reduced-Form Model](https://github.com/NatLabRockies/bsm-public-rf)
is a released application of this workflow. It distributes a fitted Biomass
Scenario Model surrogate with 65 required inputs, 245 retained features, and
23,495 outputs.

Use the repositories for different tasks:

| Task                                           | Repository      |
| ---------------------------------------------- | --------------- |
| Fit, evaluate, and export a reduced-form model | `rfm-pipeline`  |
| Run predictions with the released BSM model    | `bsm-public-rf` |

The BSM repository includes the coefficient matrix, feature definitions,
input/output metadata, prediction package, command-line interface, and small
examples. Its training data are not required for inference and are not part of
either distribution.

```bash
git clone https://github.com/NatLabRockies/bsm-public-rf.git
cd bsm-public-rf
python -m pip install .
python examples/predict.py
```

The generic synthetic example in this repository shows the fitting and export
path. The BSM example shows how a completed bundle is presented to end users
without adding domain-specific code or artifacts to the generic package.
