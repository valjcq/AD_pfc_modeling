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

#### Bump attractor behavior - debugging session

Started from observation that the bump was decreasing in amplitude over delay rather than drifting, and that distractors created a double-bump instead of shifting the bump.

Train of thought:
- Amplitude decay → network is below/near bifurcation point, recurrent excitation too weak to sustain the bump. Increased w_pyr_pyr_inter until amplitude stabilized.
- Double bump with distractor → uniform inhibition too weak to enforce winner-take-all. Two bumps coexist because neither suppresses the other.
- Bump not shifting even without distractor → in a true continuous attractor, the bump should drift spontaneously due to noise (random walk on the ring). No drift suggests either grid pinning or noise too weak.
- Tested 128, 256, 512 nodes → no change, rules out discretization artifacts.
- Tested with increased noise → still no drift, the bump is actually less stable with more noise. The bump seems to be pinned to the grid, which is surprising given the number of nodes. 

### Discovery
- The OU noise type doesn't give the difference accross condition observed in the article. However, it gives the variance of the firing rate across trials. We can do the opposite observations with the white noise.
- The code params are in better range than the supp_info one, but they are weaker value than the ones in the article.
- If we increase the current on PYR populations, all populations will increase their firing rates.
- With the fitted parameters, the model doesn't reproduces the box plots from the article. And also with the fitted parameter without VIP->VIP and PV->SOM connections. 

### Bump attractor behavior and sensitivity to parameters
The bump attractor is more stable with higher weight, and also with higher stimuli amplitude.
The main problem we run into now is that the bump created isn't really shifting in terms of angle, but decrease in amplitude. Moreover, when we introduce a distractor, the initial bump continue existing while the distractor induce a second bump, which create a multiple bump state. This is not really what we expect, as the bump should shift from the initial position to the distractor position. I think this is due to the fact that the degree of spread of our gaussian connectivity matrix is too low (10°), which make the bump really narrow and therefore more difficult to shift. I did increase it to 30°, without noticing a big change in the bump stability, but it is still not shifting with the distractor. Maybe the inhibition is not strong enough (that would also explain why we have a multiple bump state with the distractor, as the inhibition is not strong enough to suppress the initial bump/ to suppress the distractor bump). I will try to increase the inhibition and see if it can help to have a more shifting bump with the distractor.

### Rate model vs state model (transient/min vs firing rate)
- Most of the paper working on bump attractor in the context of working memory are based on firing rate model, and not changing state model.
My intuition would be that on a steady state (half awake state) the networks tend to activate really fast. In terms of state, it is really close to the bifurcation point between the two state. That's why we can observe a oscillation between two state (UP and DOWN state) as the network is in a critical state. This intuition is consistent with the observation that with perubation of the network, the occurence of this state switch is increasing whereas the amplitude of these state remind the same.
Therefore, it is important to question in the context of working memory. Does incoming stimuli would push the network in a monostable state, or increasing the frequencies of the switch? To my opinion, it would be the first case, but in this setting, our model fitted to the frequency of state switching wouldn't be well suited for simulating a working memory task.

I understood that the model in itslef (wilson cowen based equation) is suited to represent evolution of firing rate over time. So i think in our case it would be better to fit it on the frequency data and not the transient/min data. (However, we would lack the timing information but this is also the case when fitting to transient/min.). That said, the model won't behave differently with transient data or frequency data. However, it will be more acceptable to compare our results to the actual litterature on bump attractor.


### Compte et Al connectivity matrix
I realize that their definition of the connectivity matrix makes so that PYR population can have a negative input on further PYR population, which is very unlikely in the brain (no PYR TO PYR inhibition). I think we shouldn't use their definition, anyway it looks alike the gaussian connectivity matrix used in our model so far. But I should check to articles to see if it's a common definition in the litterature or if it's a specific choice of Compte et Al. I hope others article use just a gaussian connectivity matrix.
However, i changed the degree of spread of our gaussian to 30° instead of 10°. See in the bump attractor behavior and sensitivity to parameters section for why. (it's also the degree of spread used in the Compte article)


### TODO
- Dig on how they do a real test in working memory (appart from testing the stability of the bump attractor) E.G by trying a readout of the bump attractor activity and see if it can be used to decode the stimulus value after a delay period.
- Explore the litterature on bump attractor in the context of working memory, and see their metrics and how the fit their weights and stimuli amplitude.
- Think about how to read each populations impact on the bump attractor stability and metrics. 
- Try to put a distractor in the model and see how it impact the bump attractor stability and metrics/ readout capacity.