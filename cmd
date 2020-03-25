
python inference.py --config ./saved/DANet/02-03_14-52/config.json --model ./saved/DANet/02-03_14-52/checkpoint-epoch100.pth --images /home/ubuntu/BBQ/r_val

saved/LEDNet/02-03_16-15/checkpoint-epoch100.pth 
python inference.py --config ./saved/DenseASPP/02-03_16-26/config.json --model ./saved/DenseASPP/02-03_16-26/best_model.pth --images /home/ubuntu/BBQ/r_val

python inference.py --config ./saved/Test/02-07_04-37/config.json --model ./saved/Test/02-07_04-37/checkpoint-epoch150.pth --images /home/ubuntu/BBQ/val

/home/ubuntu/pytorch_segmentation-master/saved/Test/02-07_04-37
python train.py --config config.json -f ./saved/HRNetV2_OCR_GDL/02-05_07-59/checkpoint-epoch150.pth

/home/ubuntu/BBQ/r_train
/home/ubuntu/BBQ/r_trainval
/home/ubuntu/BBQ/r_val

/home/ubuntu/FYP-Seg/saved/TM2-BiSeNet/02-29_10-33
python inference.py --config ./saved/TM2-BiSeNet/02-29_10-33/config.json --model ./saved/TM2-BiSeNet/02-29_10-33/best_model.pth --images /home/ubuntu/TM2/test

python train.py --config config.json

50

/home/ubuntu/FYP-Seg/saved/TM2-HRNetV2_OCR/03-01_07-05
python inference.py --config ./saved/TM2-HRNetV2_OCR/03-01_07-05/config.json --model ./saved/TM2-HRNetV2_OCR/03-01_07-05/checkpoint-epoch200.pth --images /home/ubuntu/TM2/test

/home/ubuntu/FYP-Seg/saved/TM2-HRNetV2_OCR_Nearest/03-09_18-06/best_model.pth
python inference.py --config ./saved/TM2-HRNetV2_OCR_Nearest/03-09_18-06/config.json --model ./saved/TM2-HRNetV2_OCR_Nearest/03-09_18-06/checkpoint-epoch250.pth --images /home/ubuntu/TM2/test
python convertToONNX.py 2>&1 | tee screen.txt