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

### Discovery
- The OU noise type doesn't give the difference accross condition observed in the article. However, it gives the variance of the firing rate across trials. We can do the opposite observations with the white noise.
- The code params are in better range than the supp_info one, but they are weaker value than the ones in the article.
- If we increase the current on PYR populations, all populations will increase their firing rates.
- With the fitted parameters, the model doesn't reproduces the box plots from the article. And also with the fitted parameter without VIP->VIP and PV->SOM connections. 

### Bump attractor
- Most of the paper working on bump attractor in the context of working memory are based on firing rate model, and not changing state model.
My intuition would be that on a steady state (half awake state) the networks tend to activate really fast. In terms of state, it is really close to the bifurcation point between the two state. That's why we can observe a oscillation between two state (UP and DOWN state) as the network is in a critical state. This intuition is consistent with the observation that with perubation of the network, the occurence of this state switch is increasing whereas the amplitude of these state remind the same.
Therefore, it is important to question in the context of working memory. Does incoming stimuli would push the network in a monostable state, or increasing the frequencies of the switch? To my opinion, it would be the first case, but in this setting, our model fitted to the frequency of state switching wouldn't be well suited for simulating a working memory task.

I understood that the model in itslef (wilson cowen based equation) is suited to represent evolution of firing rate over time. So i think in our case it would be better to fit it on the frequency data and not the transient/min data. (However, we would lack the timing information but this is also the case when fitting to transient/min.). That said, the model won't behave differently with transient data or frequency data. However, it will be more acceptable to compare our results to the actual litterature on bump attractor.

### TODO
- Try to fit the model with membrane time constant set at 20ms.
- Check the 2 state model to see why this one give time-depending frequency and switch between multiple state.