# tensor analysis studio

tensor algebra with einstein notation, index tracking, metrics, coordinate transforms, and visualizations

## what it does

- einstein notation with automatic contractions
- covariant/contravariant index tracking
- metric tensors for raising/lowering indices
- coordinate system transformations (cartesian, spherical, cylindrical)
- christoffel symbols and connections
- tensor field visualizations
- general relativity examples

## install

```bash
# clone and install
git clone https://github.com/yourusername/tensoranalysis-studio
cd tensoranalysis-studio
pip install -e .
```

## run examples

```bash
# verify installation
python verify_installation.py

# run general relativity examples
python examples/general_relativity_basics.py

# open jupyter notebook with all features
jupyter notebook examples/complete_tutorial.ipynb
```

## quick usage

```python
import numpy as np
from tas import Tensor
from tas.core.metrics import euclidean_metric
from tas.core.einsum_parser import einsum_eval

# create tensors with indices (^i = upper, _j = lower)
A = Tensor(np.array([[1, 2], [3, 4]]), indices=("^i", "_j"))
B = Tensor(np.array([[5, 6], [7, 8]]), indices=("^j", "_k"))

# matrix multiply using einstein notation
C = einsum_eval("A^i_j B^j_k", A=A, B=B)

# use metrics to raise/lower indices
g = euclidean_metric(3)
v = Tensor(np.array([1, 2, 3]), indices=("_i",))
v_up = g.raise_index(v, axis=0)  # convert to contravariant
```

## coordinate transformations

```python
from tas.core.coords import SphericalFrame

# convert between coordinate systems
spherical = SphericalFrame()
cartesian_point = np.array([1.0, 1.0, 1.414])
spherical_point = spherical.from_cartesian(cartesian_point)

# get metric in different coordinates
g_spherical = spherical.metric(spherical_point)
```

## what to run

1. **verify_installation.py** - check everything works
2. **examples/general_relativity_basics.py** - see schwarzschild metric, christoffel symbols
3. **examples/complete_tutorial.ipynb** - full interactive tutorial with visualizations

## files structure

```
tas/
├── core/
│   ├── tensor.py          # main tensor class
│   ├── indices.py         # index tracking
│   ├── metrics.py         # metric tensors
│   ├── einsum_parser.py   # einstein notation
│   ├── algebra.py         # symmetrize, trace, etc
│   ├── coords.py          # coordinate systems
│   └── connections.py     # christoffel symbols
├── viz/                   # visualization tools
└── sym/                   # symbolic integration (future)

examples/
├── complete_tutorial.ipynb           # jupyter notebook with all features
└── general_relativity_basics.py     # GR examples

tests/                     # test suite
```

## requirements

- python >= 3.9
- numpy >= 1.24.0
- matplotlib (for visualizations)

## license

mit

## questions

open an issue on github or check the examples/
  year = {2025},
  url = {https://github.com/yourusername/tensor-analysis-studio}
## questions

open an issue on github or check the examples/