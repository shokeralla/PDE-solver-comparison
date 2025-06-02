| Metric | FEM (Adaptive FEM) | HMC (Bayesian Inference) |
|--------|-------------------|------------------------|
| Avg. MSE | 3.2×10⁻⁵ | 1.1×10⁻³ |
| Training Runtime (hrs) | 0.5 | 4.7 |
| Inference Runtime (sec) | 5-30 | 60-150 |
| Memory Usage (GB) | 2.1 | 8.3 |

**Notes:**
- FEM inference runtime typically ranges from a few seconds to tens of seconds based on computational complexity and model size
- HMC inference runtime can vary significantly based on dataset size, with typical values in the range of 1-2.5 minutes for standard problems, though it can extend to hours for large datasets
