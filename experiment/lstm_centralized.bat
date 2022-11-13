@echo off
setlocal enabledelayedexpansion

cd ../lstm_model
set seed=10827216

python build_field.py ../dataset/emotion.csv
python ../dataset/split_dataset.py ../dataset/emotion.csv -t --destination-folder=../dataset/centralized_data --seed=%seed%

set dir_name[0]=lr01
set dir_name[1]=lr005
set dir_name[2]=lr001
set dir_name[3]=lr0005
set dir_name[4]=lr0001
set dir_name[5]=lr00005

set lr[0]=0.01
set lr[1]=0.005
set lr[2]=0.001
set lr[3]=0.0005
set lr[4]=0.0001
set lr[5]=0.00005

for /l %%n in (0,1,5) do (
    mkdir ..\experiment\centralized_result\!dir_name[%%n]!
    python train_centralized_model.py ../dataset/centralized_data/train.csv ../dataset/centralized_data/valid.csv --saving-directory=../experiment/centralized_result/!dir_name[%%n]! --epoch=2 --learning-rate=!lr[%%n]!
    echo learning rate: !lr[%%n]!>>../experiment/centralized_result/test_result.txt
    echo ----- training result ----->>../experiment/centralized_result/test_result.txt
    python test_model.py ../experiment/centralized_result/!dir_name[%%n]!/model.pt ../dataset/centralized_data/train.csv >>../experiment/centralized_result/test_result.txt
    echo ----- testing result ----->>../experiment/centralized_result/test_result.txt
    python test_model.py ../experiment/centralized_result/!dir_name[%%n]!/model.pt ../dataset/centralized_data/test.csv >>../experiment/centralized_result/test_result.txt
    echo:>>../experiment/centralized_result/test_result.txt
    python draw_loss_chart.py ../experiment/centralized_result/!dir_name[%%n]!/accuracy_metrics.pt --saving-path=../experiment/centralized_result/accuracy_!dir_name[%%n]!.png --y-axis-str=Accuracy
    python draw_loss_chart.py ../experiment/centralized_result/!dir_name[%%n]!/loss_metrics.pt --saving-path=../experiment/centralized_result/loss_!dir_name[%%n]!.png --y-axis-str=Loss
)
