
<h1 align="center">

![image](https://user-images.githubusercontent.com/109565405/180415297-a0a6208b-da05-4c42-ac63-80987bb7ba50.png)
<br>
</h1>

<h1 align="center">
  <br>
Customer_Segmentation

<br>

<h4 align="center"><a>
created by Nurul Fathihah  <br>
July 2022
</a></h4>

<h4 align="center"><a>

![Python](https://img.shields.io/badge/Version-Python%203-006799.svg) 

</a></h4>

## Summary and Motivation

The purpose of this project is to predict the outcome of the campaign in persuade more clients to deposit money into the bank based on customers' demographic and last campaign details etc. This project was classified under classification problem and been solved by the deep learning approach.

- Research Questions
1. What are the accuracy of the model 
2. What are the best features used for this classification model

## Libraries that been used

- matplotlib.pyplot 
- numpy
- pandas
- os
- sklearn
- datetime
- tensorflow/tensorboard

## Summary of Results

![image](https://user-images.githubusercontent.com/109565405/180421388-beb7d964-c1be-49da-aa26-d7d00bdddc9b.png)

The best features used for this model are:
1)customer_age
2)balance
3)day_of_month
4)last_contact_duration 
5)num_contacts_in_campaign
6)days_since_prev_campaign_contact 
7)num_contacts_prev_campaign 

and The target for this model: term_deposit_subscribed 

![image](https://user-images.githubusercontent.com/109565405/180422352-df504b01-09e3-4d52-9729-9e836f4c95e5.png)
![model_result](https://user-images.githubusercontent.com/109565405/180422384-55ab11cf-1617-4d97-8d03-9ef69ab34662.PNG)

Model was trained with batch_size = 128, and epochs = 50 and stop at 14 epoch with the used of early_callback
The model achieve accuracy at 90% and loss 23.48%.
 
Training and validation loss and accuracy were plotted.
![epoch_acc](https://user-images.githubusercontent.com/109565405/180422446-b3c6fa29-29aa-4cff-ab46-d43bcf6e8a4a.png)
![epoch_loss](https://user-images.githubusercontent.com/109565405/180422474-cfe99916-9c0e-457f-ac5a-0b0dfe40da6e.png)

The model of epoch accuracy and loss also displayed at tensorboard
![epoch_acc_tensor](https://user-images.githubusercontent.com/109565405/180422603-72be2253-386f-4515-ba48-d0d0e449e80c.PNG)
![epoch_loss_tensor](https://user-images.githubusercontent.com/109565405/180422624-af1924f6-2375-43fd-8b5a-a21bcd1982c2.PNG)

The Model architecture were plotted in this project
![model](https://user-images.githubusercontent.com/109565405/180423635-2544d8df-b753-4e5e-96fb-449c5b93312b.png)

-Model Evaluation

![model_result](https://user-images.githubusercontent.com/109565405/180420180-7b4e9564-da45-4443-896e-a6f72f174aa1.PNG)
![confusion_matrix_display](https://user-images.githubusercontent.com/109565405/180420221-9c350209-ca49-477c-b234-24a3e4b7155a.png)

The results of this analysis: The accuracy obtained for this classification model is around 0.9 where about 8285 clients is correctly predicted tends to not deposit
money into banks based on some factors such as demographic profile and etc.


## Credits
Thanks to [HackerEarth HackLive: Customer Segmentation | Kaggle](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon). for contributions of customer segmentation dataset and my lecturer, Dr Warren for his sharing


