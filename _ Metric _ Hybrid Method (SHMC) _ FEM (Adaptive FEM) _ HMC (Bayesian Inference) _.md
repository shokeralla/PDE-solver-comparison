| Metric | Hybrid Method (SHMC) | FEM (Adaptive FEM) | HMC (Bayesian Inference) |
|-------|----------------------|-------------------|------------------------|
| Inference Time (sec) | 15-45 | 5-30 | 60-150 |
| Accuracy (MSE) | 2.8×10⁻⁴ | 3.2×10⁻⁵ | 1.1×10⁻³ |
| Memory Usage (GB) | 3.5 | 2.1 | 8.3 |
| Training Time (hrs) | 1.2 | 0.5 | 4.7 |
| Sampling Efficiency | O(N^(1/4)) faster than HMC | Medium | Low for complex systems |

**Notes:**
- The Hybrid Method (SHMC) achieves an O(N^(1/4)) speedup compared to HMC by sampling from all of phase space using high-order approximations
- FEM inference time typically ranges from a few seconds to tens of seconds based on computational complexity and model size
- HMC inference time can vary significantly based on dataset size, with typical values in the range of 1-2.5 minutes for standard problems
- SHMC requires extra storage and modest computational overhead compared to HMC, but achieves an order of magnitude higher sampling efficiency
- The acceptance rate of HMC decreases exponentially with increasing system size or time step, while SHMC achieves higher acceptance rates
