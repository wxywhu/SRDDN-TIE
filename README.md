# SRDDN-TIE
The implementation of the following paper in Pytorch
```
Ma J, Wang X, Jiang J. Image Super-Resolution via Dense Discriminative Network[J]. IEEE Transactions on Industrial Electronics, 2019.
```
## Dependencies
```
Python 3.6  PyTorch >= 1.0.0  numpy  skimage  imageio  matplotlib  tqdm
```
## Training data
```
1.Download DIV2K training data (800 training + 100 validtion images) from https://data.vision.ee.ethz.ch/cvl/DIV2K/
2.Specify '--dir_data' based on the HR and LR images path. in option.py, 
	        '--ext' is set as 'sep_reset', which first convert .png to .npy
```   
## For training
```
Cd to 'TrainCode/code
for example
CUDA_VISIBLE_DEVICES=0  python main.py --model SRDDN --save SRDDNX2 --scale 2  --n_resblocks 12 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96

you can change the settings found in option.py
```
## For testing 
```
Download the pre-trained model(SRDDN/x2/x3/x4) in https://pan.baidu.com/s/15sMaYQ3ODUZfwW3jf1aHCA  
password：wau0  
Put it in 'TestCode/model' file.
Cd to 'TestCode/code'  
run python main.py --data_test MyImage --scale 4 --model SRDDN --n_resblocks 12 --n_feats 64 --pre_train ../model/model_x4.pt --test_only --save_results --chop --save 'SRDDNX4' --testpath ../LR/LRBI --testset Set5  you can change the settings found in option.py  
Then you can find the SR result in 'TestCode/SR'. Run 'Evaluate_PSNR_SSIM.m'in MATLAB to obtain PSNR/SSIM values for paper

```
