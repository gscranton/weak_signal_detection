Launch a sweep with launch_all_sweeps.py. Settings for each data folder number should be edited within that script.

The script launch_all_sweeps.py will call several instances of run_all_sweeps.py in different terminal tabs. This in tern calls functions from runge_kutta_fourth_ordre_coupled_oscillators.py

After a sweep is done, view the results by calling sweep_analysis.py. See script for commands to call it with correct parameters for each data folder.

Additional scripts:
make_noise.py: generate Gaussian noise
testpulse.py: load and plot pulse from file (not included here)
sweep_analysis_multirun.py: variation on sweep_analysis.py that loads from multiple data folders

