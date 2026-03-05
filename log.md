### 06.02
- Exploration of the parameter set and simulations testing.
- Build two parameter sets (code and info_supp) : the one written on the code and the one coming from the supp info

### 09.02
- Checked the data from the paper and fit the model to it. (Redo the optimization with the data from the paper)
- Launch the fitting on a set without the weight that are not observed in the article (VIP->VIP and PV -> SOM)
- Run a analysis of the impact of the current on each population, with differents metrics, with all the possible parameters set obtained so far.

### 10.02
- Did a recap of the results obtained so far and the next steps to do.

### 11.02
- Started to explore the litterature about bump attractor and the parameter fit.
- Thought about the implications of fitting a model on change of state (transients/min) instead of firing rate in the context of working memory.

### 12.02 
- Explored the Comte article about working memory and bump attractor with interneurons. The model is based on a firing rate model, and not a rate model. There's no distinction of the different type of interneurons, but this work is interessant to base the working memory task on, and to check their connections value.
- Did some simulation of the ring with multiples values of the stimuli and accross differents conditions.
The model do exactly what we expect, with a bump of activity that is more or less strong depending on the value of the stimulus. On top of that, the bump metrics are different across conditions, with a stronger bump for the WT condition and a weaker bump for the APP condition. The KO condition also alter the stabilit of the bump, which is consistent with our hypothesis that the nAChR dysfunction would alter the stability of the bump attractor and therefore the working memory performance.

### 16.02
- Slides + Presentation of results and next steps.

### 17.02
- Went over Seeholzer et al. 2019 article to understand how they quantify the drift with a distractor. Decided to use their methods because it allows to quantify the drift accross different conditions.
- Did the implementation and debugging, with readme updated.
- Explored the optimization parameter for stimuli and weight inter nodes to have a bump attractor stable, with set $A^hat$ value according to noise evaluated level.

### 18.02
- Continue the work on the optimization of the parameters for the stimuli and weights inter nodes.
- Verification of the connectivity matrice from Compte et Al. -> See the corresponding point below.
- Investigate a lot of parameter set for stimuli and weights inter nodes. Also the value of the inhibition matrix.

### 19.02
- Seminar day with EI learning and Pasteur seminar.
- The article from Wimmer et Al. 2014 doesn't give a lot of information on the parameters fitting process, i try their code and it doesn't show a strong shifting with noise only.

### 20.02
- Changed the code structure to run on the GPU. 
- Run the full calibration with multiple conditions + changed the plotting of the calibration to compare accross conditions.
- TODO: Create a summary of the steps (fitting models on data, calibration of the bump attractor, pick the parameters set, try the shifting with differents conditions, Analysis of the population impact on the bump attractor behavior and accross conditions, etc) and the next steps to do.
- Explored a bit the litterature and the difference between decaying bump and bistable bump attractor, and the implications of each one on our experiment.

### 23.02
- Continue the work on the calibration of the bump attractor, with multiple conditions and multiple parameter sets.
- Read the article from Chen et Al 2024 about synaptic ring. It can give intersting way to shift the bump attractor with oriented stimuli.
- Created new metrics to analyze each population on the bump formation and stability, with the idea to see how each population contribute to the bump attractor behavior and how it can be altered in the different conditions.
- Updated the slides with new results and next steps.

### 24.02
- Tried to implement an other paper bump attractor from scratch. The idea was to have a shifting behavior with a firing rate model, and then try to adapt with our model. WIP

### 25.02
- Updated the slides with the higher inhibition parameter set, developped the figures specially in the distractor sweep experiment.
- Analyse the data from the distractor sweep experiment, trying to understand the impact of the condition on the behavior with distractor.

### 26.02
- Developped the animated figure to observe the bump attractor in 3D, over time. -> see that the bump vary over space and time in WT-APP condition wether the bump is more stable in WT condition.
- TODO: Need to write analysis down.
- Change the noise floor experiment to have a better visualization of the difference between conditions in the presence of noise (without cue or distractor). The WT-APP is slightly shifted, as we think that the network state should be in the edge of the change of state, this switch should lead the network to be more sensitive to noise, and that might be related with the bump attractor instability over space in the WT-APP condition.
- Send a mail to Boris to ask for a meeting.

### 02.03
- Continue the slides development.
- Create the assymetry experiment and analyze the results, fine tuned the graphic for the presentation.
- Adapt the MP4 formation to have faster mp4 creation.
- ADD the per population metric, start to analyze the impact of each population on the bump attractor behavior and stability, and how it can be important in the oscillation pattern and the working memory performance.
- Explore the litterature about the oscillation pattern in the context of working memory to see if the oscilation we see in our models is a feature that can be observed in the litterature, or if it is a consequence of the parameters set we have. 

### 03.03
- Fine tuned the asymmetry experiment, with more robust metrics and better visualization.
- Continue the slides with mostly the asymmetry part.


### 04.03
- Realize a bug in the burn-in of the noise in the model, which was not correctly implemented. This bug can have a strong impact on the results, especially for the noise floor experiment, and the stability of the bump attractor in the different conditions. I will need to rerun all the experiments with the corrected code, and update the figures and analysis accordingly. The overall conclusions shouldn't change much, except for the asymettry in pre cue period for example.
- TODO: A comprehensive analysis of the impact of each population on the bump attractor behavior and stability, why is there a oscillations, what's driving it, how does this oscillations change accross conditions, and with distractor.

### 05.03
- Created the `ring-asymmetry-amp-sweep` experiment: sweeps cue amplitude (default 20–60×) across conditions and measures how delay-period asymmetry (mean|A(t)| and std(A)) evolves with stimulus strength.
- Key design: one shared 6000 ms burn-in per condition (computed once), then a 1000 ms per-trial secondary burn-in from that shared state, then cue + delay. This amortises the expensive burn-in across the amplitude sweep.
- Cache is fully interchangeable with `ring-asymmetry`: both commands read/write the same per-amplitude `asymmetry_trials.csv` directories, so trials accumulated by either command are reused by the other.
- New outputs: `asymmetry_amp_sweep.png` (mean±SEM + OLS fit with slope and R² per condition) and `asymmetry_amp_sweep_violin.png` (full distributions per amplitude), plus a summary CSV and console regression/Mann-Whitney U table.
- Usage: `python -m circuit_model ring-asymmetry-amp-sweep --w_pyr_pyr_inter 7 --w_pv_global 10 --conditions WT WT_APP --amplitudes 20 30 40 45 50 60 --n_trials 50 --no_show`

### TODO
- Think about how to read each populations impact on the bump attractor stability and metrics.
- Do a decision summary (talking about the bump, why ours is a decaying bumpo, there's not strong conscencus about that and so on) 
- Do a decision summary about the "why we choose these parameters set" (why this inhibition value with these excitation value, -> related to the Pyr activity during working memory, )
- Fix the extrem firing rate in noise-floor and calibrate. (Amp is 0 with the cap at 200Hz, need to remove properly the point and )