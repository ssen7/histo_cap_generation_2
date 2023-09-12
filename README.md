To run the lightning scripts, use requirements.txt to install the necessary packages and then run the training scripts. Details on the different script are given bellow.

- data_files/ directory: Contains the necessary vocabulary file (word_map.json) and train/test/val pandas dataframe stored as a pickle file to preserve data types.
- models.py: Contains model definitions
- dataloader.py: Contains the custom PyTorch dataloader.
- patch_4k_h5.py: Contains the code for patching high resolution WSIs in the form SVS images and save the patches in an hdf5 format.
- generate4k_256clsreps.py: Extract the representations from pre-trained ViT. WARNING: To run this script download the following GitHub repo: https://github.com/mahmoodlab/HIPT


### Training scripts:

- Train vanilla resnet model: training_script_only_resnet.py and training_script_only_resnet_random_init.py (self explanatory).
- Train model with only ViT encoder: training_script_only_vit.py
- Train model with ViT encoder + Resnet reps: training_script_only_resnet_plus_vit.py