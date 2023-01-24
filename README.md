# SpikingJellyNN
This is the final project of the Computer Data Scientist Master. This is developed by SpikingJelly repo and after try their examples I started with a new dataset with 19 animals to test the library.

If you want to run DVS Gesture or MNNIST dataset you need to install the environment: 
> conda env create -f environment11classes.yml
> conda activate spikingjelly11

If you want to execute with SL-Animals you need to install the another environment:
> conda env create -f environment19classes.yml
> conda activate spikingjelly

To execute both, you need to download the datasets and put in datasets folder. A example to run the code is in the code folder and run:
> python -m clock_driven.examples.classify_dvsg -T 16 -device cuda:0 -b 6 -epochs 10 -data_dir datasets/SL-animals/ -amp -cupy -opt Adam -lr 0.001 -j 8 -out_dir datasets/SL-animals/

This code is based on: 
- Spikingjelly: https://github.com/fangwei123456/spikingjelly/tree/0.0.0.0.12/spikingjelly
- SNN-RNN: https://github.com/hewh16/SNNs-RNNs
- DCLL: https://github.com/nmi-lab/dcll

To generate a real test dataset I used:
https://github.com/SensorsINI/v2e

If you want to see the results you can find in:
https://drive.google.com/drive/folders/1Aw3eRkY8T04lb5Uyehp-EWvhp4uC2dAM?usp=sharing
