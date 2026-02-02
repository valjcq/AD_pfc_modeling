# Plan

## Recent edits summary:
- Organized the files into 2 folders: `2_state_code` and `mean_steady_code`. The first one is the old code that detect transitions states, the second one is the new code that computes the average frequencies and fit the model to experimental steady-state data.
- Created the plotting module in `mean_steady_code` to handle a single run with parameter given simuluation and plotting.

## Next steps:
### Double-checking:
- Check the experimental data availability with Boris.
- Re-run the fitting procedure to check the results that are in the original paper.
- Create a summary with the averaged frequencies from experimental data and the fitted model, with the corresponding fitted parameters.
### Exploration/New analyses:
- Try to change the different inputs, are they multiples ? Is it one general input ? See the impact of changing the input on the frequencies with the fitted model.
- Create a summary about the sensitivity of the network on the inputs.
### Ring-architecture:
- Create a ring architecture model and see with a basic input on one node how the signal propagates.