# VDMN_U-net-ESN
VDMN, U-net and ESN

Add these files into your indegrated development environment (I used pycharm myself)
for both the VDMN and the U-net you're going to have to make sure to install the nvidia-cuda so that the gpu can be used instead of the cpu.  If you want to switch between the cpu and gpu there are lines in the code that can be easily commented out.  

INSTRUCTIONS FOR THE U-net AND ESN (use the diffusion modules, unet_ml_project, and utils in one project folder, requirements.txt for this one)
Create a folder in your home directory that will be used as a shared volume between your linux machine and the docker container:
Example: create a folder called UNetMLProject in your home dir, like this = /home/(yourUSERNAMEhere)/UNetMLProject

Go to https://seungjunnah.github.io/Datasets/reds.html (If you're going to use a different dataset then make sure to have it in the right pathing for your training and testing)

Download the following datasets from the Google Drive: train_sharp, val_sharp, train_sharp_bicubic, val_sharp_bicubic

Create train and validation (val) folders within /home/marc/UNetMLProject like this:
/home/marc/UNetMLProject/train
/home/marc/UNetMLProject/val

Place train_sharp and train_sharp_bicubic zip files in /home/(yourUSERNAMEhere)/UNetMLProject/train and then unzip those zip files in that directory
Then place val_sharp and val_sharp_bicubic zip files in /home/(yourUSERNAMEhere)/UNetMLProject/val and then unzip those files in that directory as well.
After updating the pathing for the given files and using the requirements.txt to install your requirements then run the model by running the 


The same process of updating the links that path to your images.

The breakdown of the code is included in my paper so if you need a breakdown of what does what the paper should be a guide to helping understand the code.
