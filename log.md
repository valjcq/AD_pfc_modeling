### 06.02
- Exploration of the parameter set and simulations testing.
- Build two parameter sets (code and info_supp) : the one written on the code and the one coming from the supp info

### 09.02
- Checked the data from the paper and fit the model to it. (Redo the optimization with the data from the paper)
- Launch the fitting on a set without the weight that are not observed in the article (VIP->VIP and PV -> SOM)
- Run a analysis of the impact of the current on each population, with differents metrics, with the optimized parameters and the code defaults parameters.



### Discovery
- The OU noise type doesn't give the difference accross condition observed in the article. However, it gives the variance of the firing rate across trials. We can do the opposite observations with the white noise.
- The code params are in better range than the supp_info one, but they are weaker value than the ones in the article.
- If we increase the current on PYR populations, all populations will increase their firing rates.

### TODO
- Try to fit the model with membrane time constent set at 20ms.