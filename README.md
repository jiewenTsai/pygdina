# pyGDINA: Pure-Python Cognitive Diagnosis Modeling

`pyGDINA` is a lightweight, pure-Python implementation of the **Generalized Deterministic Inputs Noisy And gate (GDINA)** model. It is designed to be a Python-native alternative for researchers who currently use the `GDINA` or `CDM` packages in R.

## Key Features

- **Design Alignment:** Matches R conventions for attribute patterns (binary counting order) and initialization.
- **Fast Estimation:** Uses Marginal Maximum Likelihood with EM (MMLE/EM).
- **Flexible:** Supports all CDMs expressible under the GDINA framework (e.g., DINA, DINO, A-CDM).
- **Multi-start Support:** Includes a deterministic start (compatible with R CDM) and random starts to find the best global solution.

## Installation

Simply download `pygdina.py` and place it in your project folder, or import it directly into Google Colab.

Python

```
import pandas as pd
from pygdina import GDINA

# Initialize model
model = GDINA(att_dist="saturated", n_starts=3)
```

## Usage Example (sim30GDINA)

This example uses data structures based on the `sim30GDINA` dataset from the R GDINA package, which includes 1,000 examinees and a $30 \times 5$ Q-matrix.

Python

```
# Assuming you have your data and Q-matrix as numpy arrays
# dat: (1000, 30) response matrix
# Q:   (30, 5) Q-matrix

model.fit(dat, Q)

# Get Mastery Probabilities (mp) - similar to personparm(fit, "mp") in R
mp = model.person_parm("mp")
df_mp = pd.DataFrame(mp, columns=['A1', 'A2', 'A3', 'A4', 'A5'])

# Round to 4 decimal places for clean output
print(df_mp.round(4).head())
```

## References

This implementation is based on the following works:

- **de la Torre, J. (2011).** The generalized DINA model framework. *Psychometrika*.
- **Ma, W., & de la Torre, J. (2020).** GDINA: An R Package for Cognitive Diagnosis Modeling. *Journal of Statistical Software*.
- **Robitzsch, A., et al. (2023).** CDM: Cognitive Diagnosis Modeling (R Package).

