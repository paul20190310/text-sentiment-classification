@echo off
setlocal enabledelayedexpansion

cd ../lstm_model
set seed=10827216

python build_field.py ../dataset/emotion.csv
python ../dataset/split_dataset.py ../dataset/emotion_5050_1.csv -t --destination-folder=../dataset/partition_5050_data/p1 --seed=%seed%
python ../dataset/split_dataset.py ../dataset/emotion_5050_2.csv -t --destination-folder=../dataset/partition_5050_data/p2 --seed=%seed%

set dir_name[0]=lr02
set dir_name[1]=lr015
set dir_name[2]=lr01
set dir_name[3]=lr005
set dir_name[4]=lr0025
set dir_name[5]=lr001
set dir_name[6]=lr0005
set dir_name[7]=lr0001

set lr[0]=0.02
set lr[1]=0.015
set lr[2]=0.01
set lr[3]=0.005
set lr[4]=0.0025
set lr[5]=0.001
set lr[6]=0.0005
set lr[7]=0.0001

for /l %%n in (0,1,7) do (
  mkdir ..\experiment\federated_5050_result\!dir_name[%%n]! ..\experiment\federated_5050_result\!dir_name[%%n]!\client_1 ..\experiment\federated_5050_result\!dir_name[%%n]!\client_2

  python simulate_fed.py "python train_federated_server.py --local-epoch=1 --num-round=200 --learning-rate=!lr[%%n]! --saving-directory=../experiment/federated_5050_result/!dir_name[%%n]!" "python train_federated_client.py ../dataset/partition_5050_data/p1/train.csv ../dataset/partition_5050_data/p1/valid.csv ../dataset/partition_5050_data/p1/test.csv --saving-directory=../experiment/federated_5050_result/!dir_name[%%n]!/client_1" "python train_federated_client.py ../dataset/partition_5050_data/p2/train.csv ../dataset/partition_5050_data/p2/valid.csv ../dataset/partition_5050_data/p2/test.csv --saving-directory=../experiment/federated_5050_result/!dir_name[%%n]!/client_2"
  
  python test_model.py ../experiment/federated_5050_result/!dir_name[%%n]!/server_model.pt ../dataset/emotion_extra.csv >../experiment/federated_5050_result/!dir_name[%%n]!/test_result.txt
  python draw_loss_chart.py ../experiment/federated_5050_result/!dir_name[%%n]!/server_metrics.pt --saving-path=../experiment/federated_5050_result/!dir_name[%%n]!/loss_chart.png --x-axis-str="Rounds"
)
