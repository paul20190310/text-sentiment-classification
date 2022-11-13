@echo off
setlocal enabledelayedexpansion

cd ../lstm_model
set seed=10827216

python build_field.py ../dataset/emotion.csv
python ../dataset/partition_dataset.py ../dataset/emotion.csv -r 0.8 0.2 -f emotion_8020_1.csv emotion_8020_2.csv --seed=%seed% --destination-folder=../dataset/federated_8020_data
python ../dataset/split_dataset.py ../dataset/federated_8020_data/emotion_8020_1.csv -t --destination-folder=../dataset/federated_8020_data/data_1 --seed=%seed%
python ../dataset/split_dataset.py ../dataset/federated_8020_data/emotion_8020_2.csv -t --destination-folder=../dataset/federated_8020_data/data_2 --seed=%seed%

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
    mkdir ..\experiment\federated_8020_result\!dir_name[%%n]! ..\experiment\federated_8020_result\!dir_name[%%n]!\client_1 ..\experiment\federated_8020_result\!dir_name[%%n]!\client_2
    python simulate_fed.py "python train_federated_server.py --local-epoch=1 --num-round=200 --learning-rate=!lr[%%n]! --saving-directory=../experiment/federated_8020_result/!dir_name[%%n]!" "python train_federated_client.py ../dataset/federated_8020_data/data_1/train.csv ../dataset/federated_8020_data/data_1/valid.csv ../dataset/federated_8020_data/data_1/test.csv --saving-directory=../experiment/federated_8020_result/!dir_name[%%n]!/client_1" "python train_federated_client.py ../dataset/federated_8020_data/data_2/train.csv ../dataset/federated_8020_data/data_2/valid.csv ../dataset/federated_8020_data/data_2/test.csv --saving-directory=../experiment/federated_8020_result/!dir_name[%%n]!/client_2"
    echo learning rate: !lr[%%n]!>>../experiment/federated_8020_result/test_result.txt
    python test_model.py ../experiment/federated_8020_result/!dir_name[%%n]!/server_model.pt ../dataset/emotion_extra.csv >>../experiment/federated_8020_result/test_result.txt
    echo:>>../experiment/federated_8020_result/test_result.txt
    python draw_loss_chart.py ../experiment/federated_8020_result/!dir_name[%%n]!/server_accuracy_metrics.pt --saving-path=../experiment/federated_8020_result/accuracy_!dir_name[%%n]!.png --x-axis-str=Epochs --y-axis-str=Accuracy
    python draw_loss_chart.py ../experiment/federated_8020_result/!dir_name[%%n]!/server_loss_metrics.pt --saving-path=../experiment/federated_8020_result/loss_!dir_name[%%n]!.png --x-axis-str=Epochs --y-axis-str=Loss
)
