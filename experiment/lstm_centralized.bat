@echo off
setlocal enabledelayedexpansion

cd ../lstm_model
set seed=10827216

python build_field.py ../dataset/emotion.csv
python ../dataset/split_dataset.py ../dataset/emotion.csv -t --destination-folder=../dataset/centralized_data --seed=%seed%

set dir_name[0]=lr02
set dir_name[1]=lr0175
set dir_name[2]=lr015
set dir_name[3]=lr0125
set dir_name[4]=lr01
set dir_name[5]=lr005
set dir_name[6]=lr0025
set dir_name[7]=lr001
set dir_name[8]=lr0005
set dir_name[9]=lr0001

set lr[0]=0.02
set lr[1]=0.0175
set lr[2]=0.015
set lr[3]=0.0125
set lr[4]=0.01
set lr[5]=0.005
set lr[6]=0.0025
set lr[7]=0.001
set lr[8]=0.0005
set lr[9]=0.0001

for /l %%n in (0,1,9) do (
    mkdir ..\experiment\centralized_result\!dir_name[%%n]!
    python train_centralized_model.py ../dataset/centralized_data/train.csv ../dataset/centralized_data/valid.csv --saving-directory=../experiment/centralized_result/!dir_name[%%n]! --epoch=200 --learning-rate=!lr[%%n]!
    python test_model.py ../experiment/centralized_result/!dir_name[%%n]!/model.pt ../dataset/centralized_data/test.csv >../experiment/centralized_result/!dir_name[%%n]!/test_result.txt
    python draw_loss_chart.py ../experiment/centralized_result/!dir_name[%%n]!/metrics.pt --saving-path=../experiment/centralized_result/!dir_name[%%n]!/loss_chart.png
)
