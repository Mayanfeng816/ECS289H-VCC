# Enhancing Visual Concept Connectome with Adaptive Feature Granularity

## Steps to run this code
1. Download dataset:
https://www.kaggle.com/datasets/arjunashok33/miniimagenet
2. Clone the code repository and place the dataset into the project directory dataset folder.
3. run `batch_run_adaptive_vcc.py` and `batch_run_original_vcc.py` to get the output into the outputs folder. You can change the number of reading class based on your requiremnt
4. run folder  `exps` corresponding experiments (`experiment1_granularity.py`,`experiment2_concept_stability.py`,`experiment3_connectivity.py`) to get corresponding cache in `cache` folder
5. run `merge_and_plot.py` with suffix `--exp 1` or you change the number to 2 or 3 based on your requirement to plot the corresponding experiment, the plot will output in the `exp_outputs` folder and relevant value will be in the `exp_outputs/csv` folder

## Related resources
original VCC

https://github.com/YorkUCVIL/VCC