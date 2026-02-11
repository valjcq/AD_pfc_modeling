# Literature Review: Bump Attractor Models for Working Memory
## Parameter Optimization and Task Design

Based on your preregistration and current work on nAChR-modulated PFC networks, here's a comprehensive guide to literature and approaches for parameter optimization in bump attractor working memory tasks.

---

## 1. FOUNDATIONAL PAPERS: WANG & COMPTE MODELS

### Core Reference Papers

**Compte et al. (2000)** - "Synaptic mechanisms and network dynamics underlying visuospatial working memory in a cortical network model"
- *Cerebral Cortex*, 10: 910-923
- **Why essential**: This is THE foundational paper for bump attractor working memory models with PYR-PYR excitation and inhibition
- **Key parameters**: They establish the baseline Mexican-hat connectivity (local excitation, lateral inhibition)
- **Working memory task**: Includes delay period, distractor presentation, and measures bump stability
- **Parameter regime**: Documents the transition between sustainable and non-sustainable states

**Wang (2002)** - "Probabilistic decision making by slow reverberation in neocortical circuits"
- *Neuron*, 36
- **Key contribution**: Shows how NMDA-mediated slow recurrent excitation creates stable attractors
- **Biological constraint**: Emphasizes the importance of slow time constants (NMDA) for working memory stability

**Brunel & Wang (2001)** - Working memory spiking network model
- **Critical finding**: Shows that distractor strength must be carefully controlled - above a certain strength, the persistent memory trace is perturbed
- **Your relevance**: Directly addresses how to set parameters between sustainable and non-sustainable states

---

## 2. PARAMETER OPTIMIZATION APPROACHES

### Theoretical Framework

**Seeholzer et al. (2017)** - "Efficient low-dimensional approximation of continuous attractor networks"
- *arXiv:1711.08032*
- **Method**: Low-dimensional parametrization of firing rate profiles
- **Tool**: Provides efficient numerical optimization for network parameters
- **Application**: Can predict effects of network parameters on steady-state firing rate profiles and bump existence
- **Your use case**: Use this to fine-tune your network to produce bumps of desired shape WITHOUT time-consuming full simulations

**Key approach**: They reduce high-dimensional network dynamics to tractable self-consistent equations, making parameter optimization feasible.

### Stability and Perturbation Studies

**Laing & Chow (2001)** + **Gutkin et al. (2001)** - Short-term plasticity and stability
- **Finding**: A transient excitatory stimulus matching the memory trace location can extinguish the persistent state
- **Mechanism**: Works by transiently synchronizing spike times
- **Your application**: Design perturbations that test bump stability

**Kilpatrick et al. (2019)** - "Stability of working memory in continuous attractor networks under short-term plasticity"
- *PLOS Computational Biology*
- **Theory**: Derives analytical expressions for diffusion coefficients and drift fields
- **Parameters**: Depends on short-term plasticity parameters, firing rate profile shape, and neuron model
- **Key finding**: Without synaptic depression (τx → 0), bump states are NOT stable at low firing rates
- **Your relevance**: Critical for understanding the parameter regime between stable and unstable bumps

---

## 3. BIOLOGICAL PLAUSIBILITY & PARAMETER RANGES

### Recent Cholinergic Modulation Study

**Qi et al. (2021) + Computational Model** - "Cholinergic Neuromodulation of Prefrontal Attractor Dynamics"
- *Journal of Neuroscience*, 2024
- **Experimental data**: Nucleus basalis stimulation effects on PFC working memory
- **Model parameters**: Provides recent parameter values for biologically realistic bump attractor networks
- **Key finding**: Network depolarization reduces bump diffusion when neurons have saturating responses
- **Parameter range**: Baseline firing rates ~5-10 Hz (matching your experimental constraints)

### Parameter Ranges from Wimmer et al. (2014)

**Wimmer et al. (2014)** - "Bump attractor dynamics in prefrontal cortex explains behavioral precision"
- *Nature Neuroscience*, nn.3645
- **Experimental validation**: Uses monkey electrophysiology to constrain model parameters
- **Measures**: Bump diffusion, variability correlations, Fano factors
- **Biological constraints**: Provides experimentally-validated parameter ranges for PFC working memory

---

## 4. WORKING MEMORY TASK DESIGN

### Standard Delay-Response Task Structure

**From Compte et al. (2000) and Brunel & Wang (2001)**:

```
1. CUE PERIOD (200-500 ms)
   - Present transient stimulus
   - Amplitude: Strong enough to initiate bump (typically 15-25 Hz above baseline)
   - Should reliably form a stable bump

2. DELAY PERIOD (1-3 seconds)
   - No external input
   - Bump maintained by recurrent excitation
   - Measure: Bump position, amplitude, diffusion
   
3. OPTIONAL: DISTRACTOR PERIOD (during delay)
   - Present second stimulus at different location
   - Timing: Mid-delay (t = 1-2s after cue)
   - Amplitude: Variable (this is your key manipulation)
   - Distance: Variable angular distance from cue
   
4. RESPONSE PERIOD
   - Read out bump location
   - Compare to original cue location
   - Measure: Accuracy, precision
```

### Key Parameters to Manipulate

Based on Compte et al. (2000) and subsequent studies:

**Cue Parameters**:
- **Amplitude**: 10-30 Hz increase above baseline
- **Duration**: 200-500 ms
- **Width**: 30-60° for ring attractor networks

**Delay Parameters**:
- **Duration**: 1-3 seconds (variable to test decay)
- **Noise level**: Gaussian noise with σ = 1-5 Hz

**Distractor Parameters** (critical for your experiments):
- **Amplitude relative to cue**: 50%, 75%, 100%, 125%
- **Angular distance**: 45°, 90°, 135°, 180°
- **Timing**: Early (0.5s), mid (1.5s), late (2.5s) in delay
- **Duration**: 200-500 ms

---

## 5. DISTRACTOR RESISTANCE & BUMP STABILITY

### Critical Papers on Distractor Effects

**Compte et al. (2000)** - Original distractor experiments
- **Finding**: Network can be completely distracted by strong stimulus OR show vector averaging
- **Parameter dependence**: Depends on distractor strength and angular distance
- **Winner-take-all vs averaging**: Function of recurrent excitation strength

**Bouchacourt & Buschman (2019)** + **Wei et al. (2012)** - Multi-item capacity
- **Finding**: Limited capacity arises from attractor competition
- **Relevance**: Helps understand how network parameters determine whether distractors destroy or shift bumps

**Gutkin et al. (2001)** - Erasure mechanisms
- **Alternative to inhibition**: Excitatory stimulus matching memory location can erase bump
- **Mechanism**: Spike synchronization

### Correlation-Based Control

**Gutkin et al. (2013)** - "Correlations in background activity control persistent state stability"
- *Frontiers in Computational Neuroscience*
- **Novel mechanism**: Background activity correlations control stability
- **Application**: High correlations make sustained state unstable
- **Your use case**: Could modulate correlation levels to control bump stability without changing other parameters
- **Task implementation**: Modulate correlations at different task phases (encoding, maintenance, distraction)

---

## 6. CRITICAL PARAMETER REGIMES

### Bistability Requirements

From multiple sources (Compte, Wang, Brunel):

**For Bistable Regime** (resting state + bump state):
- Strong enough recurrent excitation to sustain bump
- Balanced inhibition to prevent runaway excitation
- Mexican-hat connectivity profile

**Parameter Constraints**:
```
J_EE (PYR→PYR): Strong locally, weak globally
J_EI (PYR→PV): Sufficient to prevent hyperactivity
J_IE (PV→PYR): Strong enough for competition, not so strong to kill bumps
```

### Boundary Between Sustainable and Non-Sustainable

**From Kilpatrick et al. (2019)**:
- **Without short-term depression**: Bump state not stable at low firing rates
- **Critical parameter**: Synaptic depression time constant τ_x
- **Bifurcation point**: Where system transitions from monostable (resting only) to bistable (resting + bump)

**Your experimental goal**: Set parameters near this bifurcation point so that:
1. Baseline conditions: Bump is marginally stable
2. Perturbations (nAChR knockout, distractors): Push system across bifurcation
3. Result: Measurable failure of maintenance

---

## 7. SPECIFIC RECOMMENDATIONS FOR YOUR WORK

### Building on Your Current Model

From your LOG.md and preregistration:

**Current Status**:
- You have PYR→PYR excitation drive
- PV interneurons provide inhibition (global?)
- Parameters from Koukouli et al. (2025) supplementary info
- Fitting challenges with observed firing rates

### Recommended Workflow

**Step 1: Establish Parameter Regime** (using Seeholzer et al. 2017 approach)
```python
# Use low-dimensional optimization to find parameter set where:
# 1. Resting state: ~5 Hz (PYR), ~10 Hz (PV) - match Koukouli data
# 2. Bump state: ~20-25 Hz peak, ~40-60° width
# 3. Transition: Cue of 20 Hz, 300 ms reliably initiates bump
# 4. Stability: Bump survives 2s delay with <10° drift
```

**Step 2: Test Bump Formation and Maintenance**
```python
# Protocol 1: Vary cue amplitude
cue_amplitudes = [10, 15, 20, 25, 30] # Hz above baseline
for amp in cue_amplitudes:
    present_cue(amplitude=amp, duration=300ms)
    delay_period(duration=2000ms)
    measure_bump_formation_probability()
    measure_bump_drift()
```

**Step 3: Test Distractor Susceptibility**
```python
# Protocol 2: Vary distractor strength (Brunel & Wang approach)
distractor_strengths = [0.5, 0.75, 1.0, 1.25, 1.5]  # relative to cue
angular_distances = [45, 90, 135, 180]  # degrees

for strength in distractor_strengths:
    for distance in angular_distances:
        present_cue(amplitude=20Hz, duration=300ms)
        delay_period(duration=1000ms)
        present_distractor(
            amplitude=strength*20Hz,
            duration=300ms,
            angular_distance=distance
        )
        delay_period(duration=1000ms)
        measure_bump_location()
        measure_bump_amplitude()
        calculate_distraction_effect()
```

**Step 4: Test nAChR Modulation Effects**

Based on Koukouli et al. (2025) experimental findings:
```python
conditions = ['WT', 'alpha7_KO', 'beta2_KO', 'APP']

for condition in conditions:
    set_nAChR_parameters(condition)
    
    # Test 1: Baseline bump stability
    measure_spontaneous_bump_drift()
    
    # Test 2: Cue-evoked bump formation
    test_cue_amplitude_threshold()
    
    # Test 3: Distractor resistance
    test_distractor_susceptibility()
    
    # Test 4: Bump decay over extended delays
    test_delay_duration_limits(max_delay=5000ms)
```

### Key Metrics to Track

**From Wimmer et al. (2014) + Compte et al. (2000)**:

1. **Bump Position Accuracy**: Angular error between final and cued location
2. **Bump Position Precision**: Trial-to-trial variability (circular std dev)
3. **Bump Amplitude**: Peak firing rate during maintenance
4. **Bump Width**: Full-width at half-maximum (FWHM)
5. **Diffusion Coefficient**: Rate of random drift over time
6. **Distraction Bias**: Systematic shift toward distractor location
7. **Bump Collapse Probability**: Fraction of trials where bump disappears

---

## 8. SPECIFIC PARAMETER VALUES FROM LITERATURE

### From Compte et al. (2000) - Ring Attractor Model

**Network Size**: 
- 512 neurons arranged in ring
- Preferred direction uniformly distributed [0, 2π]

**Connectivity**:
- W(θ) = J+ exp(-α|θ|²) - J- (Mexican hat)
- J+ ≈ 2.0-2.5 (local excitation strength)
- J- ≈ 0.5-1.0 (global inhibition strength)
- α controls width of excitation

**Firing Rates**:
- Resting: 3-5 Hz
- Bump peak: 20-30 Hz
- Background: 5-10 Hz

**Time Constants**:
- τ_AMPA = 2 ms
- τ_NMDA = 100 ms (critical for stability!)
- τ_GABA = 10 ms

### From Wang (2002) - Decision Circuit

**Synaptic Weights** (relative to baseline):
- w+ (within selective pool): 1.7-2.1
- w- (between pools): 0.6-0.9

**External Input**:
- Baseline: I_ext = 0.3 nA
- Stimulus: ΔI = 0.01-0.05 nA (weak for slow integration)

### From Qi et al. (2021) / Recent Model (2024)

**Firing Rate Ranges** (from experiments):
- PYR baseline: 4.8 Hz (baseline) → 10 Hz (with NB stimulation)
- Experimental data: 10.9 → 13.4 Hz

**Key Finding for Parameter Setting**:
- Neurons need saturating responses for stability improvement with depolarization
- Linear regime: Depolarization doesn't help
- Saturating regime: Depolarization reduces diffusion

---

## 9. NOISE AND PERTURBATION TYPES

### From Short-Term Plasticity Paper (Kilpatrick et al. 2019)

**Types of Noise**:
1. **Poisson spiking noise**: Inherent in spike generation
2. **Background input noise**: σ_noise = 0.5-2.0 Hz
3. **Heterogeneity**: Cell-to-cell variability in parameters (10-20% CV)

### From Your LOG.md Discovery

Your observation:
- OU noise: Doesn't give difference across conditions but gives variance across trials
- White noise: Gives opposite observation

**Recommendation**: 
- Use **both** noise types for different purposes
- OU noise: For testing trial-to-trial variability (behavioral relevance)
- White noise: For testing condition differences (biological mechanism)

---

## 10. EXPERIMENTAL PREDICTIONS FROM MODELS

### From Wimmer et al. (2014) - Validated Predictions

Their model predicted (and confirmed experimentally):
1. Neural variability correlates with behavioral imprecision
2. Neurons with flank stimuli show activity correlations with behavior
3. Fano factors follow specific patterns
4. Noise correlations depend on stimulus position

**Your application**: Use similar readouts to validate your model:
- Variability in bump position → variability in behavior
- Cell-to-cell correlations follow bump structure
- nAChR dysfunction → changes in specific predicted patterns

---

## 11. SOFTWARE AND IMPLEMENTATION

### Recommended Tools

**NEST Simulator**: 
- For large-scale spiking networks
- Already used in Koukouli et al. model

**Brian2**:
- More flexible for rapid parameter exploration
- Better for testing different network architectures

**PyNN**:
- Cross-platform specification
- Used in several bump attractor studies

### Parameter Optimization Strategies

**From Koukouli et al. (2025) Supplementary**:
- Differential evolution (global optimization)
- Followed by Nelder-Mead (local refinement)
- Error function: Mean absolute percentage error across all targets

**Your challenge** (from LOG.md):
- Current params don't reproduce boxplots from article
- Need to try: τ_membrane = 20 ms instead of current value
- Consider KO conditions for fitting

---

## 12. KEY PAPERS SUMMARY TABLE

| Paper | Year | Key Contribution | Your Relevance |
|-------|------|------------------|----------------|
| Compte et al. | 2000 | Foundational bump attractor model | Base architecture, distractor protocols |
| Wang | 2002 | NMDA-based slow reverberation | Time constant requirements |
| Brunel & Wang | 2001 | Distractor strength control | Parameter regime for failure modes |
| Wimmer et al. | 2014 | Experimental validation | Parameter ranges, validation metrics |
| Seeholzer et al. | 2017 | Efficient optimization method | Your parameter search strategy |
| Kilpatrick et al. | 2019 | Stability theory with STP | Understanding sustainable/non-sustainable boundary |
| Gutkin et al. | 2013 | Correlation-based control | Alternative perturbation mechanism |
| Qi et al. | 2021/2024 | Cholinergic modulation | Recent parameters, NB stimulation effects |

---

## 13. IMPLEMENTATION CHECKLIST

### Phase 1: Model Validation (2-3 weeks)
- [ ] Implement ring attractor architecture
- [ ] Use Seeholzer et al. method for parameter optimization
- [ ] Match baseline firing rates to Koukouli et al. data
- [ ] Verify bump formation with transient cue
- [ ] Measure bump stability over 2-3s delays
- [ ] Validate bump width (40-60°) and peak amplitude (20-30 Hz)

### Phase 2: Basic Working Memory Task (2-3 weeks)
- [ ] Implement cue-delay-response protocol
- [ ] Vary cue amplitude to find threshold
- [ ] Vary delay duration to measure decay
- [ ] Add noise (both OU and white) at appropriate levels
- [ ] Measure drift/diffusion coefficients
- [ ] Compare to Wimmer et al. metrics

### Phase 3: Distractor Resistance (3-4 weeks)
- [ ] Implement distractor presentation
- [ ] Vary distractor strength (0.5-1.5x cue)
- [ ] Vary angular distance (45-180°)
- [ ] Vary distractor timing (early/mid/late delay)
- [ ] Measure: bump shift, collapse probability, vector averaging
- [ ] Compare to Compte et al. predictions

### Phase 4: nAChR Modulation (3-4 weeks)
- [ ] Implement WT, α7-KO, β2-KO, APP conditions
- [ ] Test bump formation thresholds across conditions
- [ ] Test distractor resistance across conditions
- [ ] Test delay duration limits across conditions
- [ ] Measure all metrics from Wimmer et al. for each condition
- [ ] Generate predictions for experimental validation

---

## 14. CRITICAL QUESTIONS TO ADDRESS

### Theoretical Questions

1. **What parameter regime ensures marginally stable bumps?**
   - Too stable: Insensitive to perturbations
   - Too unstable: Spontaneous collapse even without perturbations
   - Answer: From Kilpatrick et al. - near bifurcation point, with τ_x tuned appropriately

2. **How do nAChR changes affect this regime?**
   - α7-KO: Changes in PYR excitability (Koukouli data: reduced Vm fluctuations)
   - β2-KO: Changes in high-affinity cholinergic current
   - Prediction: Should shift bifurcation point

3. **What metrics best predict cognitive impairment?**
   - Bump position accuracy?
   - Bump position precision?
   - Collapse probability?
   - Answer: Combination, but precision (trial-to-trial variability) best correlates with behavioral WM (Wimmer et al.)

### Practical Questions

1. **What distractor strength separates conditions?**
   - Use Brunel & Wang approach: Find strength where WT succeeds but KO fails
   - Typically: 75-100% of cue strength

2. **What delay duration reveals deficits?**
   - Short delays (0.5-1s): All conditions succeed
   - Medium delays (2-3s): KO conditions show increased drift
   - Long delays (4-5s): KO conditions show collapse

3. **What angular distance is most informative?**
   - Near distractors (45-90°): Test vector averaging
   - Far distractors (135-180°): Test winner-take-all

---

## 15. ADDITIONAL RESOURCES

### Textbooks

1. **"Neuronal Dynamics" by Gerstner et al.** - Chapter 18.3 on Bump Attractors
   - Online: neuronaldynamics.epfl.ch
   - Clear mathematical treatment

2. **"Theoretical Neuroscience" by Dayan & Abbott**
   - Attractor networks chapter
   - Ring models for head direction

### Code Repositories

1. **Wang Lab Publications** - www.cns.nyu.edu/wanglab
   - Original model code often available

2. **ModelDB** - senselab.med.yale.edu/modeldb
   - Search for Compte (2000) model #: 7354
   - Search for Wang (2002) models

### Review Articles

1. **Wang (2001)** - "Synaptic reverberation underlying mnemonic persistent activity"
   - *Trends in Neurosciences*
   - Excellent overview of mechanisms

2. **Rolls (2010)** - "Attractor networks"
   - *Wiley Interdisciplinary Reviews*
   - Broader context of attractor dynamics

---

## CONCLUSION

Your project sits at an exciting intersection of:
1. **Biophysically detailed** nAChR modulation (Koukouli et al.)
2. **Computational theory** of bump attractors (Compte, Wang)
3. **Cognitive relevance** to Alzheimer's disease

### Key Recommendations:

1. **Start with Seeholzer et al. (2017) optimization method**
   - Will save you weeks of parameter search
   - Provides mathematical framework for understanding parameter effects

2. **Use Compte et al. (2000) task structure**
   - Well-validated
   - Clear predictions
   - Direct comparison to experiments

3. **Follow Brunel & Wang (2001) for distractor strength**
   - Carefully control distractor amplitude
   - This is where your biological manipulations will show effects

4. **Validate with Wimmer et al. (2014) metrics**
   - These connect neural dynamics to behavior
   - Necessary for translational relevance to AD

5. **Leverage Kilpatrick et al. (2019) stability theory**
   - Understand exactly where your network sits in parameter space
   - Predict when perturbations will cause failures

### Expected Timeline:
- **Months 1-2**: Parameter optimization and basic WM task
- **Month 3**: Distractor susceptibility characterization
- **Month 4**: nAChR condition comparisons
- **Month 5**: Analysis and manuscript preparation

Good luck with your experiments!
