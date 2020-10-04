import os

# Parameters
dataroot = r"C:/Users/arkha/Desktop/Study/Intern/DAAD_Project/to_share"
name = 'tum_pix2pix'
model = 'pix2pix_tum'
direction = 'AtoB'
input_nc = 1
output_nc = 1
dataset_mode = 'tum'
local = True
gpu_ids = '-1'

# Command Line
os.system("C:/Users/arkha/Anaconda3/envs/TUM/python.exe run.py --dataroot {} --name {} --model {} --direction {} --input_nc {} --output_nc {} --dataset_mode {} --local {} --gpu_ids {}".format(dataroot, name, model, direction, input_nc, output_nc, dataset_mode, local, gpu_ids))


