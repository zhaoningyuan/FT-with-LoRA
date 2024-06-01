# FT-with-LoRA
## Setup
We recommend using the latest release of [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch).

First, pull the Docker image (please replace `xx.xx` with the actual version number):
``` bash
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/FT-with-LoRA:/workspace/FT-with-LoRA -v /path/to/models:/models nvcr.io/nvidia/pytorch:xx.xx-py3
```
Install Requirements
``` bash
cd FT-with-LoRA
pip install -r requirements.txt
```
## Usage
After installation, there are several possible workflows. The most comprehensive is:  

1. Data download and preprocessing   
2. Finetuning with LoRA  
3. Downstream task evaluation or text generation  
4. Plot results  
### Download Data and Preprocessing
Download pCLUE data: 
``` bash
cd data
bash downloadpClUE.sh
```
Preprocess the data: 
``` bash
cd data
bash preprocess.sh
```
### Finetune
Run the following command to finetune: 
``` bash
bash run_peft_ds.sh
```
### Evaluation
Run the following command to evaluate: 
``` bash
bash eval.sh
```
### Plot results
Run the following command to plot the results:  
``` bash
python plotResult.py
```


