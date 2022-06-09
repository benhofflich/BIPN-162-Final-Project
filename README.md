# BIPN-162-Final-Project

## Replication Paper:
MoDL: Model Based Deep Learning Architecture for Inverse Problems
https://arxiv.org/pdf/1712.02862.pdf 

### GitHub:
https://github.com/hkaggarwal/modl 

## Addtional Dataset:
A dual autoencoder and singular value decomposition based feature optimization for the segmentation of brain tumor from MRI images
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8117624/ 

### Github:
https://github.com/guillaumefrd/brain-tumor-mri-dataset/blob/master/images/slices_example.png 

## Files in this GitHub:
`modl-master` : the repository from the original paper's GitHub. contains the below files as well as the `savedModels` folder that couldn't be uploaded to this GitHub's main directory. The `savedModels` folder contains the trained models that were created by running `trn.py`.

`demoImage.hdf5` : the testing image used in `testDemo.py` for image reconstruction.

`model.py` : contains code for creating the CNN model and the conjugate gradient algorithm.

`supportingFunctions.py` : contains supporting functions for calculating the time and PSNR values, and reading the dataset. This also contains functions that we added so that we could read in the additional dataset.

`testDemo.py` : the file for testing the trained model against the testing image. This was used for replicating the paper's results. 

`tstDemo.ipynb` : the file for testing the trained model against the testing image. This was used to test the additional dataset.

`trn.py` : the file for training the model. It will save the trained model as a folder inside the `savedModels` folder. This is the main file that we will manipulating. 

`dataset.hdf5` : the file is too large to upload to GitHub so it has to be downloaded via [this link](https://drive.google.com/file/d/1qp-l9kJbRfQU1W5wCjOQZi7I3T6jwA37/view?usp=sharing). Contains the training dataset for `trn.py`. 

## How to Run
- `savedModels`, `demoImage.hdf5`, `model.py`, `supportingFunctions.py`, `testDemo.py`, `trn.py`, `dataset.hdf5` `tstDemo.ipynb` must be in the same directory
- `trn.py` must be run on Google Colab
- First run `trn.py` with the training dataset. After the model is trained, it will save in the `savedModels` folder and be named according to the date and time it was run. 
- To replicate the paper's results, run `testDemo.py`. To run the additional dataset, run `tstDemo.ipynb`.
