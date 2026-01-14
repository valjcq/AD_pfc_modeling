tau_a_1 = 0.6  # adaptation time for pyramidal population, in s.
tau_a_3 = 0.6  # adaptation time for somatostatin interneuron population, in s.
sigma = 0.0192  # noise level

fact_pyr = 344.0  # scale for pyramidal population firing rate activity.
fact_pv = 163.0  # scale for pyramidal population firing rate activity.
fact_som = 23657.0  # scale for pyramidal population firing rate activity.
fact_vip = 1018.0  # scale for pyramidal population firing rate activity.

# parameters for nicotinic receptors level of action.
I_ext = 0.3  # external background input current (not use ?)
nb_a7 = 1
nb_b2 = 1
nb_a5 = 1

theta_pyr = 7.0  # minimum input for the input-output relationship of PYR poulation.
alpha_pyr = 1.9  # slope for the input-output relationship of PYR poulation.
theta_inter_pv = (
    7.0  # minimum input for the input-output relationship of PV population.
)
alpha_inter_pv = 2.6  # slope for the input-output relationship of PV population.
theta_inter_som = (
    7.0  # minimum input for the input-output relationship of SOM population.
)
alpha_inter_som = 1.5  # slope for the input-output relationship of SOM population.
theta_inter_vip = (
    7.0  # minimum input for the input-output relationship of VIP population.
)
# slope for the input-output relationship of VIP population.
alpha_inter_vip = 1.2

kd = 0.9
# Non-zero connection weights:
w_11 = 30.0  # connection weight for pyr to pyr
w_21 = 30.0  # connection weight for pv to pyr
w_31 = 53.0  # connection weight for som to pyr
w_12 = 41.0  # connection weight for pyr to pv
w_22 = 33.0  # connection weight for pv to pv
w_13 = 22.0  # connection weight for pyr to som
w_43 = 49.0  # connection weight for vip to som
w_14 = 12.0  # connection weight for pyr to vip
w_34 = 35.0  # connection weight for som to vip

i_pyr = 0.9  # external excitatory input current to pyr

# Non-connected weights set to zero:
w_23 = 0.0  # connection weight for pv to som
w_24 = 0.0  # connection weight for pv to vip
w_32 = 0.0  # connection weight for som to pv
w_33 = 0.0  # connection weight for som to som
w_42 = 0.0  # connection weight for vip to pv
w_44 = 0.0  # connection weight for vip to vip

# There's different external inputs for each interneuron type?
i_inter_pv_max = 6.6  # external excitatory input current to pv
i_inter_som_max = 1.6  # external excitatory input current to som
i_inter_vip_max = 2.8  # external excitatory input current to vip

mult_factor_i = 1  # multiplicative factor for external inputs to interneurons.

J_r1 = 1.7  # parameter for pyramidal spike frequency adaptation.
