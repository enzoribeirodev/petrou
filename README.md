# petrou

**petrou** is a Python library for image thresholding and segmentation optimization. It implements Otsu, Tsallis, and MASI thresholding criteria, three optimization backends (Exhaustive Search, Simulated Annealing, and Particle Swarm Optimization), segmentation evaluation metrics, and a Bresenham line-profile tool — all sharing a single, uniform interface built around the `SearchSpace` abstraction.

```
pip install petrou
```

Optional visualisation dependency (required only for the `line_profile_bresenham` overlay):

```
pip install petrou[vis]
```

---

## Table of contents

1. [Package layout](#1-package-layout)
2. [Importing petrou](#2-importing-petrou)
3. [Quick start](#3-quick-start)
4. [SearchSpace](#4-searchspace)
5. [Objectives](#5-objectives)
6. [Optimizers](#6-optimizers)
   - 6.1 [Exhaustive search](#61-exhaustive-search)
   - 6.2 [Simulated Annealing](#62-simulated-annealing)
   - 6.3 [PSO](#63-pso)
   - 6.4 [InertiaRegistry](#64-inertiaregistry)
7. [Bi-level thresholding](#7-bi-level-thresholding)
   - 7.1 [ThresholdResult](#71-thresholdresult)
   - 7.2 [Otsu](#72-otsu)
   - 7.3 [Tsallis](#73-tsallis)
   - 7.4 [MASI](#74-masi)
8. [Multi-level thresholding](#8-multi-level-thresholding)
9. [Segmentation metrics](#9-segmentation-metrics)
10. [Line profile](#10-line-profile)
11. [Exceptions](#11-exceptions)
12. [Developer guide — adding a new optimizer](#12-developer-guide--adding-a-new-optimizer)
13. [Developer guide — adding a new criterion](#13-developer-guide--adding-a-new-criterion)

---

## 1. Package layout

```
petrou/
├── __init__.py
├── exceptions.py
├── optimization/
│   ├── search_space.py    SearchSpace, VariableDef
│   ├── sa.py              simulated_annealing
│   ├── exhaustive.py      exhaustive_search
│   └── pso.py             PSO, InertiaRegistry
├── objectives/
│   ├── variance.py        otsu_criterion
│   └── entropy.py         tsallis_entropy, tsallis_q_automatic,
│                          masi_entropy, masi_r_adaptive
├── thresholding/
│   ├── bi_level.py        find_otsu_threshold, find_tsallis_threshold,
│   │                      find_masi_threshold, ThresholdResult
│   └── multi_level.py     multilevel_otsu, multilevel_tsallis, multilevel_masi
├── metrics/
│   └── segmentation.py    misclassification_error, false_positive_rate,
│                          false_negative_rate, jaccard_index,
│                          dice_coefficient, dice_loss, detect_gt_polarity
└── analysis/
    └── profile.py         line_profile_bresenham
```

**Layering rule:** `optimization` and `objectives` are independent of everything else. `thresholding` imports from both. `metrics` and `analysis` are standalone. Nothing imports upward.

---

## 2. Importing petrou

The recommended style is to import directly from the submodule that owns the symbol:

```python
from petrou.thresholding.bi_level import find_otsu_threshold, find_tsallis_threshold
from petrou.thresholding.multi_level import multilevel_otsu
from petrou.optimization.search_space import SearchSpace
from petrou.optimization.sa import simulated_annealing
from petrou.optimization.pso import PSO, InertiaRegistry
from petrou.objectives.variance import otsu_criterion
from petrou.objectives.entropy import tsallis_entropy, masi_entropy
from petrou.metrics.segmentation import jaccard_index, dice_coefficient
from petrou.analysis.profile import line_profile_bresenham
from petrou.exceptions import PetrouError
```

A flat API is also available — every public symbol is re-exported from the top-level package:

```python
import petrou

result = petrou.find_otsu_threshold(img)
```

---

## 3. Quick start

```python
import numpy as np
from PIL import Image

from petrou.thresholding.bi_level import (
    find_otsu_threshold,
    find_tsallis_threshold,
    find_masi_threshold,
)
from petrou.metrics.segmentation import jaccard_index, dice_coefficient

# Load a grayscale image as a NumPy array
img = np.array(Image.open("image.png").convert("L"))

# --- Otsu ---
otsu = find_otsu_threshold(img)
seg_otsu = (img > otsu.threshold).astype(np.uint8) * 255
print(f"Otsu  t={otsu.threshold}  score={otsu.score:.2f}")

# --- Tsallis (q estimated automatically) ---
tsallis = find_tsallis_threshold(img)
seg_tsallis = (img > tsallis.threshold).astype(np.uint8) * 255
print(f"Tsallis  t={tsallis.threshold}  q={tsallis.params['q']:.3f}")

# --- MASI (r estimated from image statistics) ---
masi = find_masi_threshold(img)
seg_masi = (img > masi.threshold).astype(np.uint8) * 255
print(f"MASI  t={masi.threshold}  r={masi.params['r']:.3f}")

# --- Evaluate against a ground truth ---
gt = np.array(Image.open("ground_truth.png").convert("L"))
for name, seg in [("Otsu", seg_otsu), ("Tsallis", seg_tsallis), ("MASI", seg_masi)]:
    ji = jaccard_index(seg, gt)
    dc = dice_coefficient(seg, gt)
    print(f"{name:8s}  JI={ji:.4f}  DC={dc:.4f}")
```

---

## 4. SearchSpace

`SearchSpace` is the central abstraction shared by all optimizers. It defines which variables exist, their types, their bounds, and the perturbation scale for each. Every optimizer in petrou accepts a `SearchSpace` and calls the same two methods — `initial_state` and `neighbour` — without knowing anything about images.

### Declaring a search space

```python
from petrou.optimization.search_space import SearchSpace

space = SearchSpace([
    {"name": "q",  "type": "float", "bounds": (0.01, 3.0), "step": 0.05},
    {"name": "t",  "type": "int",   "bounds": (1,    254), "step": 5},
])
```

Each variable is a dict with four required keys:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Identifier. Used as the key in `decode()` output. |
| `type` | `"float"` or `"int"` | Governs perturbation style and rounding in `decode()`. |
| `bounds` | `(lo, hi)` | Inclusive interval. `lo < hi` required. |
| `step` | `float` | Perturbation scale. For `float`: σ of Gaussian delta. For `int`: half-width of discrete uniform step. |

The list order defines the state vector indices: `state[0]` is the first variable, `state[1]` is the second, and so on.

### Internal representation and `decode`

All optimizers work with `np.ndarray` of `float64` internally, even for integer variables. `decode()` converts that raw vector back to typed Python values:

```python
rng = np.random.default_rng(0)
state = space.initial_state(rng)   # np.ndarray, e.g. [1.52, 127.0]

decoded = space.decode(state)
# {"q": 1.52, "t": 127}  — q is float, t is Python int
```

**When to use `decode()`:** call it once after the optimizer returns. Do not call it inside objective functions — it allocates a dict on every call.

```python
# Inside the objective (hot path) — inline cast, no dict
obj = lambda s: tsallis_entropy(hist, float(s[0]), int(round(s[1])))

# After the optimizer returns (cold path) — decode once
best_state, best_score = simulated_annealing(obj, space, ...)
result = space.decode(best_state)
print(result["q"], result["t"])
```

To decode a single variable without building the full dict:

```python
t = space.decode_value(best_state, "t")   # int
q = space.decode_value(best_state, "q")   # float
```

### Generating states and neighbours

```python
rng = np.random.default_rng(42)

state     = space.initial_state(rng)
neighbour = space.neighbour(state, rng, perturbation="independent")
neighbour = space.neighbour(state, rng, perturbation="single")
```

Perturbation per variable type:

- **`float`**: `δ ~ N(0, step²)`, result clipped to bounds.
- **`int`**: `δ ∈ {-step, …, -1, +1, …, +step}` (zero excluded), result clipped and rounded.

Use `"independent"` when variables are correlated. Use `"single"` for high-dimensional or independent spaces.

### Other methods

```python
space.clip(state)      # clamp a state to bounds — returns a copy
space.contains(state)  # True if every dimension is within bounds
len(space)             # number of variables (== space.ndim)
space.lower            # np.ndarray of lower bounds, shape (ndim,)
space.upper            # np.ndarray of upper bounds, shape (ndim,)
```

---

## 5. Objectives

All functions in `petrou.objectives` accept a pre-computed histogram as their first argument. The caller computes `np.histogram` once; the objective functions are called thousands of times by the optimizer without repeating that work.

**Convention:** higher return value = better threshold for every function.

### `otsu_criterion`

```python
from petrou.objectives.variance import otsu_criterion
import numpy as np

hist, _ = np.histogram(img, bins=256, range=(0, 256))

# Vectorised: returns all 256 variances in one NumPy pass
variances = otsu_criterion(hist)           # np.ndarray, shape (256,)
best_t = int(np.argmax(variances))

# Scalar: returns the variance for one threshold
score = otsu_criterion(hist, t=128)        # float
```

Both modes are consistent: `otsu_criterion(hist, t=k) == otsu_criterion(hist)[k]`.

### `tsallis_entropy`

```python
from petrou.objectives.entropy import tsallis_entropy

score = tsallis_entropy(hist, q=1.5, t=128)
score = tsallis_entropy(hist, q=1.5, t=128, add_log_noise=True)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `hist` | `ndarray (256,)` | Absolute-frequency histogram. |
| `q` | `float` | Entropic index. `q = 1` recovers Shannon (Kapur). Typical range: (0.01, 3.0). |
| `t` | `int` | Candidate threshold in [0, 255]. |
| `add_log_noise` | `bool` | Add ε = 1e-12 inside logarithms. Recommended when optimizing `q` jointly. Default `False`. |

Returns `0.0` when PA = 0 or PB = 0.

### `tsallis_q_automatic`

```python
from petrou.objectives.entropy import tsallis_q_automatic

q_opt, ratio = tsallis_q_automatic(hist, q_min=0.01, q_max=2.0, steps=200)
```

Finds the `q` that minimises S_q / S_q_max. Returns `(q_opt, ratio_min)`. The ratio is diagnostic; only `q_opt` is typically needed.

### `masi_entropy`

```python
from petrou.objectives.entropy import masi_entropy

score = masi_entropy(hist, r=0.8, t=128)
score = masi_entropy(hist, r=0.8, t=128, add_log_noise=True, verbose=True)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `r` | `float` | Shape parameter. `r = 1` → Shannon. Typical range: (0.01, 3.0). |
| `add_log_noise` | `bool` | Default `True`. MASI is more sensitive to log(0) than Tsallis. |
| `verbose` | `bool` | Emit a `RuntimeWarning` when `r` is too large for the image. |

Returns `0.0` when PA = 0, PB = 0, or log argument ≤ 0.

### `masi_r_adaptive`

```python
from petrou.objectives.entropy import masi_r_adaptive

hist, _ = np.histogram(img, bins=256, range=(0, 256))
r = masi_r_adaptive(hist, img)   # float in (0, 1]
```

Estimates `r = argmax(hist) / max(pixel_value)`.

---

## 6. Optimizers

All optimizers accept `objective_fn: Callable[[np.ndarray], float]` and `search_space: SearchSpace`. They know nothing about images or histograms.

### 6.1 Exhaustive search

```python
from petrou.optimization.exhaustive import exhaustive_search

best_t, best_score = exhaustive_search(
    objective_fn = lambda t: score_at(t),
    search_range = (1, 255),   # [lo, hi) — hi is exclusive
    maximize     = True,
)
```

Evaluates every integer in `[lo, hi)`. Guaranteed global optimum. Use for 1-D integer problems.

### 6.2 Simulated Annealing

```python
from petrou.optimization.sa import simulated_annealing
from petrou.optimization.search_space import SearchSpace
import numpy as np

space = SearchSpace([
    {"name": "q", "type": "float", "bounds": (0.01, 3.0), "step": 0.05},
    {"name": "t", "type": "int",   "bounds": (1, 254),    "step": 5},
])

best_state, best_score = simulated_annealing(
    objective_fn  = lambda s: some_score(float(s[0]), int(round(s[1]))),
    search_space  = space,
    T_init        = 100.0,
    T_min         = 1e-3,
    alpha         = 0.9,        # cooling: T <- T * alpha per chain
    markov_length = 20,         # evaluations per temperature level
    boltzmann_k   = 1.0,
    max_iter      = 1_000,
    maximize      = True,
    perturbation  = "independent",
    random_state  = 42,
)

decoded = space.decode(best_state)
print(decoded["q"], decoded["t"])
```

**With convergence history:**

```python
best_state, best_score, history = simulated_annealing(
    ..., return_history=True
)
# history: [{"iter": 20, "T": 90.0, "best": 5.3}, ...]
```

**Cooling schedule:** total evaluations ≈ `markov_length × log(T_min / T_init) / log(alpha)`, capped at `max_iter`.

### 6.3 PSO

```python
from petrou.optimization.pso import PSO
from petrou.optimization.search_space import SearchSpace

space = SearchSpace([
    {"name": "t", "type": "int", "bounds": (1, 254), "step": 5},
])

pso = PSO(
    objective_fn  = lambda s: score(int(round(s[0]))),
    num_particles = 20,
    search_space  = space,
    mode          = "max",    # "max" or "min"
    k             = 0.5,      # v_max = k * (upper - lower) / 2
    c1            = 2.0,      # cognitive coefficient
    c2            = 2.0,      # social coefficient
    seed          = 42,
)

best_pos, best_fit = pso.optimize(
    max_iterations   = 100,
    inertia_strategy = 0.5,   # float or named strategy string
)
decoded = space.decode(best_pos)
```

**With convergence history:**

```python
best_pos, best_fit, history = pso.optimize(100, return_history=True)
# history: [{"iter": 0, "best": 1234.5}, {"iter": 1, "best": 1567.8}, ...]
```

### 6.4 InertiaRegistry

Built-in inertia strategies:

| Value | Description |
|-------|-------------|
| float | Constant weight |
| `"random"` | `w = 0.5 + r/2`, r ~ Uniform(0, 1) |
| `"linearly decreasing"` | Linear decay from 0.9 to 0.4 |
| `"global-local best"` | `w_ij = 1.1 − g_ij / p_ij` |
| `"chaotic descending"` | Decreasing trend + logistic chaos |
| `"chaotic random"` | Random base + logistic chaos |

```python
from petrou.optimization.pso import InertiaRegistry

# List all available strategies
print(InertiaRegistry.list_strategies())

# Register a custom strategy
import numpy as np

@InertiaRegistry.register("sigmoid")
def sigmoid_inertia(t, max_iter, particle, g_pos, g_fit, rng):
    x = 10.0 * (t / max_iter - 0.5)
    return 1.0 / (1.0 + np.exp(x))

pso.optimize(100, inertia_strategy="sigmoid")
```

Every strategy function receives these arguments in this order:

```python
def my_strategy(
    t        : int,                 # current iteration, 0-indexed
    max_iter : int,                 # total iterations
    particle,                       # current particle (.best_position, .best_fitness)
    g_pos    : np.ndarray,          # global best position so far
    g_fit    : float,               # global best fitness so far
    rng      : np.random.Generator,
) -> float | np.ndarray:           # scalar or per-dimension weight
```

---

## 7. Bi-level thresholding

### 7.1 ThresholdResult

Every `find_*_threshold` function returns a `ThresholdResult`:

```python
@dataclass
class ThresholdResult:
    threshold : int     # optimal threshold in [0, 255]
    score     : float   # objective value at optimum (higher = better)
    params    : dict    # {"q": float}, {"r": float}, or {} for Otsu
    optimizer : str     # "exhaustive", "sa", or "pso"
```

Applying the threshold:

```python
binary_mask  = img > result.threshold              # bool
binary_image = binary_mask.astype(np.uint8) * 255  # uint8, values 0 or 255
```

### 7.2 Otsu

```python
from petrou.thresholding.bi_level import find_otsu_threshold

# Exhaustive search — always optimal for Otsu
result = find_otsu_threshold(img)

# SA
result = find_otsu_threshold(
    img,
    optimizer        = "sa",
    search_range     = (0, 255),
    optimizer_config = {
        "T_init":       80.0,
        "max_iter":     500,
        "random_state": 0,
        "step":         10,    # integer step for threshold perturbation
    },
)

# PSO
result = find_otsu_threshold(
    img,
    optimizer        = "pso",
    optimizer_config = {
        "n_particles":    20,
        "max_iterations": 100,
        "seed":           0,
    },
)

print(result.threshold)   # int
print(result.score)       # float — between-class variance at optimum
print(result.params)      # {}
print(result.optimizer)   # "exhaustive"
```

### 7.3 Tsallis

**`"automatic"` — estimate `q` from the histogram:**

```python
from petrou.thresholding.bi_level import find_tsallis_threshold

result = find_tsallis_threshold(img)
# q_strategy="automatic" and optimizer="exhaustive" by default

print(result.threshold)
print(result.params["q"])   # estimated q
print(result.score)
```

**`"fixed"` — supply `q` directly:**

```python
result = find_tsallis_threshold(img, q_strategy="fixed", q_fixed=1.5)
```

**`"optimize"` with SA — jointly optimize `q` and `t`:**

```python
result = find_tsallis_threshold(
    img,
    q_strategy       = "optimize",
    optimizer        = "sa",
    q_bounds         = (0.01, 3.0),
    q_step           = 0.05,
    add_log_noise    = True,
    optimizer_config = {
        "T_init":       80.0,
        "max_iter":     600,
        "random_state": 0,
        "t_step":       5,
    },
)
print(result.threshold, result.params["q"])
```

**`"optimize"` with PSO:**

```python
result = find_tsallis_threshold(
    img,
    q_strategy       = "optimize",
    optimizer        = "pso",
    add_log_noise    = True,
    optimizer_config = {
        "n_particles":    25,
        "max_iterations": 100,
        "seed":           0,
        "t_step":         5,
    },
)
```

**`"automatic"` with SA** — `q` is resolved first, then SA optimizes only `t`:

```python
result = find_tsallis_threshold(
    img,
    q_strategy       = "automatic",
    optimizer        = "sa",
    optimizer_config = {"T_init": 60.0, "max_iter": 400, "t_step": 5},
)
```

`"optimize"` + `"exhaustive"` raises `IncompatibleStrategyError` — a continuous variable cannot be enumerated.

### 7.4 MASI

**`"adaptive"` — estimate `r` from image statistics:**

```python
from petrou.thresholding.bi_level import find_masi_threshold

result = find_masi_threshold(img)
# r_strategy="adaptive" and optimizer="exhaustive" by default

print(result.threshold)
print(result.params["r"])
```

**`"fixed"` — supply `r` directly:**

```python
result = find_masi_threshold(img, r_strategy="fixed", r_fixed=0.8)
```

**`"optimize"` with SA:**

```python
result = find_masi_threshold(
    img,
    r_strategy       = "optimize",
    optimizer        = "sa",
    r_bounds         = (0.01, 3.0),
    r_step           = 0.05,
    optimizer_config = {
        "T_init":       80.0,
        "max_iter":     600,
        "random_state": 0,
        "t_step":       5,
    },
)
print(result.threshold, result.params["r"])
```

**`"optimize"` with PSO:**

```python
result = find_masi_threshold(
    img,
    r_strategy       = "optimize",
    optimizer        = "pso",
    optimizer_config = {
        "n_particles":    25,
        "max_iterations": 100,
        "seed":           0,
        "t_step":         5,
    },
)
```

**`"adaptive"` with SA:**

```python
result = find_masi_threshold(
    img,
    r_strategy       = "adaptive",
    optimizer        = "sa",
    optimizer_config = {"T_init": 60.0, "max_iter": 400, "t_step": 5},
)
```

---

## 8. Multi-level thresholding

Multi-level thresholding recursively applies a bi-level finder to produce `k` intensity classes. Any `k ≥ 2` is accepted. The output image replaces each pixel with the mean intensity of its class.

### Otsu

```python
from petrou.thresholding.multi_level import multilevel_otsu

segmented, info = multilevel_otsu(img, k=4)

print(info["thresholds"])   # e.g. [62, 118, 187] — sorted, k-1 values
print(segmented.shape)      # same as img.shape
print(segmented.dtype)      # float32

# With SA
segmented, info = multilevel_otsu(
    img, k=3,
    optimizer        = "sa",
    optimizer_config = {"max_iter": 300, "random_state": 0},
)
```

### Tsallis

```python
from petrou.thresholding.multi_level import multilevel_tsallis

# Default: q automatic, exhaustive
segmented, info = multilevel_tsallis(img, k=3)
print(info["thresholds"])

# Joint SA optimization of q and t
segmented, info = multilevel_tsallis(
    img,
    k                = 4,
    q_strategy       = "optimize",
    optimizer        = "sa",
    add_log_noise    = True,
    optimizer_config = {"max_iter": 400, "random_state": 0, "t_step": 5},
)

# Fixed q
segmented, info = multilevel_tsallis(img, k=3, q_strategy="fixed", q_fixed=1.5)
```

### MASI

```python
from petrou.thresholding.multi_level import multilevel_masi

# Default: r adaptive, exhaustive
segmented, info = multilevel_masi(img, k=3)
print(info["thresholds"])

# Joint PSO optimization of r and t
segmented, info = multilevel_masi(
    img,
    k                = 4,
    r_strategy       = "optimize",
    optimizer        = "pso",
    optimizer_config = {"n_particles": 20, "max_iterations": 100, "t_step": 5},
)
```

All keyword arguments accepted by `find_*_threshold` are forwarded by the multi-level wrappers.

---

## 9. Segmentation metrics

All metrics compare a binary **segmented** image against a binary **ground truth**. Convention: `pixel == 0` → background, `pixel != 0` → foreground.

```python
import numpy as np
from PIL import Image
from petrou.thresholding.bi_level import find_otsu_threshold
from petrou.metrics.segmentation import (
    misclassification_error,
    false_positive_rate,
    false_negative_rate,
    jaccard_index,
    dice_coefficient,
    dice_loss,
    detect_gt_polarity,
)

result = find_otsu_threshold(img)
seg    = (img > result.threshold).astype(np.uint8) * 255
gt     = np.array(Image.open("ground_truth.png").convert("L"))

me  = misclassification_error(seg, gt)   # [0, 1] — lower is better
fpr = false_positive_rate(seg, gt)       # [0, 1] — lower is better
fnr = false_negative_rate(seg, gt)       # [0, 1] — lower is better
ji  = jaccard_index(seg, gt)             # [0, 1] — higher is better (IoU)
dc  = dice_coefficient(seg, gt)          # [0, 1] — higher is better (F1)
dl  = dice_loss(seg, gt)                 # 1 - DC — lower is better

print(f"ME={me:.4f}  FPR={fpr:.4f}  FNR={fnr:.4f}")
print(f"JI={ji:.4f}  DC={dc:.4f}  DL={dl:.4f}")
```

### Polarity detection

Some datasets store ground-truth labels with an inverted convention (white = background). Use `detect_gt_polarity` to detect the convention automatically, or pass `invert_gt=True` directly when the polarity is known.

```python
# Automatic detection
inverted = detect_gt_polarity(seg, gt)
# True  → ground truth is inverted; use invert_gt=True
# False → ground truth uses standard convention

ji = jaccard_index(seg, gt, invert_gt=inverted)
dc = dice_coefficient(seg, gt, invert_gt=inverted)
me = misclassification_error(seg, gt, invert_gt=inverted)
```

`detect_gt_polarity` computes the Dice coefficient under both polarities and returns `True` when inversion improves the score. Verify on a few images before applying to a full dataset.

**Direct override when the polarity is known:**

```python
ji  = jaccard_index(seg, gt_inverted, invert_gt=True)
dc  = dice_coefficient(seg, gt_inverted, invert_gt=True)
me  = misclassification_error(seg, gt_inverted, invert_gt=True)
fpr = false_positive_rate(seg, gt_inverted, invert_gt=True)
fnr = false_negative_rate(seg, gt_inverted, invert_gt=True)
```

---

## 10. Line profile

```python
from petrou.analysis.profile import line_profile_bresenham

# Fixed endpoints (x, y)
intensities, vis_image, pt1, pt2 = line_profile_bresenham(
    gray_image = img,
    pt1        = (10, 50),
    pt2        = (490, 400),
)

# intensities: list[int] — pixel values along the Bresenham line
# vis_image:   np.ndarray (H, W, 3) BGR with the line in red,
#              or None if opencv-python is not installed
# pt1, pt2:   effective endpoints

print(f"Samples: {len(intensities)}")
print(f"Min: {min(intensities)}  Max: {max(intensities)}")

# Random endpoints — pass None for either or both
intensities, vis_image, pt1, pt2 = line_profile_bresenham(img)
print(f"Random line {pt1} → {pt2}: {len(intensities)} samples")
```

The intensity profile is always returned. `vis_image` requires `opencv-python`; without it, `vis_image` is `None` and a `RuntimeWarning` is emitted.

---

## 11. Exceptions

All petrou exceptions inherit from `PetrouError`:

```python
from petrou.exceptions import (
    PetrouError,                 # base — catches everything
    InvalidSearchSpaceError,     # bad variable definition
    EmptyHistogramError,         # image region has no pixels
    OptimizationError,           # objective_fn raised on initial state
    IncompatibleStrategyError,   # invalid strategy combination
)
```

| Exception | Inherits from | Raised when |
|-----------|--------------|-------------|
| `PetrouError` | `Exception` | Base class. Never raised directly. |
| `InvalidSearchSpaceError` | `ValueError` | `lo >= hi`, `step <= 0`, unknown type string, missing dict key. |
| `EmptyHistogramError` | `ValueError` | The image region contains no pixels. |
| `OptimizationError` | `RuntimeError` | The objective function raised during the first evaluation. |
| `IncompatibleStrategyError` | `ValueError` | `q_strategy="optimize"` + `optimizer="exhaustive"`; unknown optimizer name; `q_fixed=None` when required. |

```python
from petrou.exceptions import IncompatibleStrategyError, InvalidSearchSpaceError
from petrou.thresholding.bi_level import find_tsallis_threshold
from petrou.optimization.search_space import SearchSpace

try:
    find_tsallis_threshold(img, q_strategy="optimize", optimizer="exhaustive")
except IncompatibleStrategyError as e:
    print(e)

try:
    SearchSpace([{"name": "x", "type": "float", "bounds": (5.0, 1.0), "step": 0.1}])
except InvalidSearchSpaceError as e:
    print(e)
```

---

## 12. Developer guide — adding a new optimizer

This example adds **Differential Evolution**. The same pattern applies to any other population-based or trajectory-based algorithm.

### Step 1 — Create the file

```
petrou/optimization/de.py
```

### Step 2 — Implement the optimizer

Three requirements:

1. Accept `search_space: SearchSpace` as the second positional argument.
2. Use `search_space.initial_state(rng)` to initialise and `search_space.clip(state)` to enforce bounds after manual movement.
3. Return `(np.ndarray, float)` — the raw best state and its score. Never decode inside the optimizer.

```python
# petrou/optimization/de.py
from __future__ import annotations
from typing import Callable
import numpy as np
from petrou.optimization.search_space import SearchSpace

__all__ = ["differential_evolution"]


def differential_evolution(
    objective_fn    : Callable[[np.ndarray], float],
    search_space    : SearchSpace,
    *,
    population_size : int   = 20,
    F               : float = 0.8,
    CR              : float = 0.9,
    max_iter        : int   = 500,
    maximize        : bool  = True,
    random_state    : int | None = None,
    return_history  : bool  = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, list[dict]]:

    rng = np.random.default_rng(random_state)
    pop = np.stack([search_space.initial_state(rng) for _ in range(population_size)])
    scores = np.array([objective_fn(ind) for ind in pop])

    best_idx = scores.argmax() if maximize else scores.argmin()
    best_state, best_score = pop[best_idx].copy(), float(scores[best_idx])
    history: list[dict] = []

    def is_better(a: float, b: float) -> bool:
        return a > b if maximize else a < b

    for t in range(max_iter):
        for i in range(population_size):
            a, b, c = pop[rng.choice(population_size, 3, replace=False)]
            mutant = search_space.clip(a + F * (b - c))
            mask  = rng.random(search_space.ndim) < CR
            trial = np.where(mask, mutant, pop[i])
            score = float(objective_fn(trial))
            if is_better(score, scores[i]):
                pop[i], scores[i] = trial, score
                if is_better(score, best_score):
                    best_state, best_score = trial.copy(), score
        if return_history:
            history.append({"iter": t, "best": best_score})

    if return_history:
        return best_state, best_score, history
    return best_state, best_score
```

### Step 3 — Export it

In `petrou/optimization/__init__.py`:

```python
from petrou.optimization.de import differential_evolution
__all__.append("differential_evolution")
```

In `petrou/__init__.py`:

```python
from petrou.optimization.de import differential_evolution
__all__.append("differential_evolution")
```

### Step 4 — Wire into thresholding

Add one branch to `_run_optimizer` in `petrou/thresholding/bi_level.py`:

```python
if optimizer == "de":
    from petrou.optimization.de import differential_evolution
    cfg = {"maximize": True, **config}
    result = differential_evolution(objective_fn, space, **cfg)
    return result[0], result[1]
```

This single change makes the optimizer available to all bi-level finders and all multi-level wrappers, because all of them route through `_run_optimizer`.

```python
# After the four steps above, this works:
from petrou.thresholding.bi_level import find_tsallis_threshold
from petrou.thresholding.multi_level import multilevel_otsu

result = find_tsallis_threshold(
    img,
    q_strategy       = "optimize",
    optimizer        = "de",
    optimizer_config = {"population_size": 40, "max_iter": 300},
)

segmented, info = multilevel_otsu(
    img, k=4,
    optimizer        = "de",
    optimizer_config = {"population_size": 30, "max_iter": 200},
)
```

---

## 13. Developer guide — adding a new criterion

This example adds **Kapur's entropy**.

### Step 1 — Implement the objective function

Add to `petrou/objectives/entropy.py`:

```python
def kapur_entropy(hist: np.ndarray, t: int) -> float:
    """
    Kapur's entropy criterion for bi-level thresholding.

    Parameters
    ----------
    hist : np.ndarray, shape (256,)
    t : int

    Returns
    -------
    float — higher is better.
    """
    p  = hist.astype(np.float64) / hist.sum()
    PA = p[:t + 1].sum()
    PB = 1.0 - PA
    if PA == 0.0 or PB == 0.0:
        return 0.0
    pA = p[:t + 1] / PA
    pB = p[t + 1:]  / PB
    HA = float(-np.sum(pA * np.log(pA + 1e-12)))
    HB = float(-np.sum(pB * np.log(pB + 1e-12)))
    return HA + HB

__all__.append("kapur_entropy")
```

Export from `petrou/objectives/__init__.py` and `petrou/__init__.py`.

### Step 2 — Implement the threshold finder

Add to `petrou/thresholding/bi_level.py`. For a criterion with no scalar parameter, model it after `find_otsu_threshold`:

```python
from petrou.objectives.entropy import kapur_entropy
from functools import partial

def find_kapur_threshold(
    img_region       : np.ndarray,
    *,
    optimizer        : str = "exhaustive",
    search_range     : tuple[int, int] = (1, 255),
    optimizer_config : dict | None = None,
) -> ThresholdResult:
    config = dict(optimizer_config or {})
    hist   = _histogram(img_region)

    if optimizer == "exhaustive":
        best_t, best_score = exhaustive_search(partial(kapur_entropy, hist), search_range)
        return ThresholdResult(threshold=best_t, score=best_score, optimizer=optimizer)

    step  = config.pop("step", 5)
    space = _1d_space(search_range, step)
    obj   = lambda s: kapur_entropy(hist, int(round(s[0])))
    best, score = _run_optimizer(obj, space, optimizer, config)
    return ThresholdResult(
        threshold=int(space.decode(best)["t"]),
        score=score,
        optimizer=optimizer,
    )
```

Add to `__all__` in `bi_level.py`, `thresholding/__init__.py`, and `petrou/__init__.py`.

### Step 3 — Add the multi-level wrapper

Add to `petrou/thresholding/multi_level.py`:

```python
from petrou.thresholding.bi_level import find_kapur_threshold

def multilevel_kapur(
    img              : np.ndarray,
    k                : int,
    *,
    optimizer        : str = "exhaustive",
    search_range     : tuple[int, int] = (1, 255),
    optimizer_config : dict | None = None,
) -> tuple[np.ndarray, dict]:
    finder = partial(
        find_kapur_threshold,
        optimizer=optimizer,
        search_range=search_range,
        optimizer_config=optimizer_config,
    )
    return _multilevel_engine(img, k, finder)
```

`_multilevel_engine` is criterion-agnostic and requires only that `finder` accepts `img_region` and returns `ThresholdResult`.

### Step 4 — If the criterion has a scalar parameter

For criteria with a shape parameter (like `q` in Tsallis or `r` in MASI), model the finder after `find_tsallis_threshold`:

1. Add a `param_strategy` argument with `"automatic"`, `"fixed"`, and `"optimize"` options.
2. For `"automatic"`: implement an estimation function in `petrou/objectives/` and call it before the search.
3. For `"optimize"`: build a 2-D `SearchSpace([param_var, t_var])` and call `_run_optimizer`. Raise `IncompatibleStrategyError` when `optimizer="exhaustive"`.
4. For `"automatic"` and `"fixed"` with a stochastic optimizer: build a 1-D `SearchSpace([t_var])` and close over the resolved scalar in the objective lambda.