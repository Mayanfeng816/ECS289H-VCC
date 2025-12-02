# Enhancing Visual Concept Connectome with Adaptive Feature Granularity

We fixed the model  `resnet50` with `--feature_names layer1 layer2 layer3 layer4`
## Steps to run this code
We have already uploaded our cache, so it can jump to step 5
1. Download dataset:
https://www.kaggle.com/datasets/arjunashok33/miniimagenet
2. Clone the code repository and place the dataset into the project directory's dataset folder.
3. Run `batch_run_adaptive_vcc.py` and `batch_run_original_vcc.py` to get the output into the `outputs` folder. You can change the number of reading classes based on your requirements
4. Run folder  `exps` corresponding experiments (`experiment1_granularity.py`,`experiment2_concept_stability.py`,`experiment3_connectivity.py`) to get corresponding cache in `cache` folder 
5. Run `merge_and_plot.py` with suffix `--exp 1` or you change the number to 2 or 3 based on your requirement to plot the corresponding experiment, the plot will output in the `exp_outputs` folder, and the relevant value will be in the `exp_outputs/csv` folder

## Related resources
original VCC

https://github.com/YorkUCVIL/VCC
