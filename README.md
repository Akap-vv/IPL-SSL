# Semi-Supervised Learning with Interpolation and Pseudo-Labeling for Few-Label Intrusion Detection
### Overview
This paper proposes IPL-SeSL, a novel SeSL framework that synergistically integrates Interpolation and Pseudo-Labeling mechanisms to enhance IDS's performance under severe label constraints. IPL-SeSL consists of a supervised branch, a pseudo-labeling branch, and an interpolation branch. In the pseudo-labeling branch, we propose a data augmentation method specifically designed for network traffic data, which enhances the model’s robustness and generalization ability. The interpolation mechanism introduces a novel sample generation strategy that reinforces decision boundaries through geometrically meaningful feature space transformations.

### Requirements
All experiments are implemented on the PyTorch platform with an NVIDIA RTX 3090 GPU.
Python package information is summarized in requirements.txt.
### Quick start
For the packet-length sequence feature, run the following code:
``` 
python trainWRN-LBS.py 
```
For the byte sequence feature, run the following code:
``` 
python trainWRN-bytes.py 
```
#### Command options
The following parameters can be specified when using ```trainWRN-LBS.py``` or ```trainWRN-LBS.py```:
* ```--batch-size```: train batchsize
* ```--mu```: coefficient of unlabeled batch size
* ```--weight```: weight of the Interpolation branch's loss
* ```--num-classes```: number of classes
* ```--lr```: learning rate
* ```--aug-level```: strong augmentation factor
* ```--weak-level```: weak augmentation factor

