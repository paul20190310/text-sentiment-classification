@echo off
setlocal enabledelayedexpansion

cd ../lstm_model
set seed=10827216

python build_field.py ../dataset/emotion.csv
python ../dataset/split_dataset.py ../dataset/emotion_33label_1.csv -t --destination-folder=../dataset/partition_33label_data/p1 --seed=%seed%
python ../dataset/split_dataset.py ../dataset/emotion_33label_2.csv -t --destination-folder=../dataset/partition_33label_data/p2 --seed=%seed%

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
    mkdir ..\experiment\federated_33label_result\!dir_name[%%n]! ..\experiment\federated_33label_result\!dir_name[%%n]!\client_1 ..\experiment\federated_33label_result\!dir_name[%%n]!\client_2
    python simulate_fed.py "python train_federated_server.py --local-epoch=1 --num-round=200 --learning-rate=!lr[%%n]! --saving-directory=../experiment/federated_33label_result/!dir_name[%%n]!" "python train_federated_client.py ../dataset/partition_33label_data/p1/train.csv ../dataset/partition_33label_data/p1/valid.csv ../dataset/partition_33label_data/p1/test.csv --saving-directory=../experiment/federated_33label_result/!dir_name[%%n]!/client_1" "python train_federated_client.py ../dataset/partition_33label_data/p2/train.csv ../dataset/partition_33label_data/p2/valid.csv ../dataset/partition_33label_data/p2/test.csv --saving-directory=../experiment/federated_33label_result/!dir_name[%%n]!/client_2"
    python test_model.py ../experiment/federated_33label_result/!dir_name[%%n]!/server_model.pt ../dataset/emotion_extra.csv >../experiment/federated_33label_result/!dir_name[%%n]!/test_result.txt
    python draw_loss_chart.py ../experiment/federated_33label_result/!dir_name[%%n]!/server_metrics.pt --saving-path=../experiment/federated_33label_result/!dir_name[%%n]!/loss_chart.png --x-axis-str="Rounds"
)