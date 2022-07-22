
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

## ABOUT THE PROJECT

Bank's revenue is mainly decrement of the deposition of money into the bank by the clients.The bank decided to conduct marketing campaigns to persuade more 
clients to deposit money into the bank. The main purpose of this project is to predict the outcome of the campaign in persuade more clients to deposit money into the bank based on customers' demographic and last campaign details etc. This project was classified under classification problem and been solved by the deep learning approach.The dataset contains  customer demographic and marketing campaign details.


## Libraries that been used

- matplotlib.pyplot 
- numpy
- pandas
- os
- sklearn
- datetime
- tensorflow/tensorboard

## Summary of Results

- Data Loading
The dataset is imported using os and pandas libraries with CSV file.

- Data Inspection 
There are 18 columns at the beginning of the dataset and two columns: id and prev_campaign_outcome were dropped due to the unnecessary to the model

In order to get better insights of the dataset: Categorical Columns were plotted against Target:term_deposit_subscribed

For the term_deposit_subscribed, we can see that there is class imbalance issue  
![comparison_chart_term_deposit_subscribed](https://user-images.githubusercontent.com/109565405/180430428-45e29683-dc25-48e2-9008-b5c3613b37f0.png)

![comparison_chart_marital](https://user-images.githubusercontent.com/109565405/180430602-3c2158c4-bd44-49a6-883b-9493b8a9085f.png)  
![comparison_chart_education](https://user-images.githubusercontent.com/109565405/180430789-d6971614-edfe-4ba3-94e0-3a92b4362132.png)

From the chart,married couple tends to have higher chance of no putting deposit in banks maybe due to many life commitments or low education level. 
While,clients who have secondary education have higher possibilties in put deposit in banks.   
  
- Data Cleaning  
There are Nan values and were treat with KNNImputer,  no duplicates data and no outliers removed due to the data conditions.

- Features Selection
Cramers V,  and Logistic Regression were used for the features selection.
  
![image](https://user-images.githubusercontent.com/109565405/180421388-beb7d964-c1be-49da-aa26-d7d00bdddc9b.png)

The best features used for this model are:
1)customer_age
2)balance
3)day_of_month
4)last_contact_duration 
5)num_contacts_in_campaign
6)days_since_prev_campaign_contact 
7)num_contacts_prev_campaign 

and the target for this model: term_deposit_subscribed 

![image](https://user-images.githubusercontent.com/109565405/180422352-df504b01-09e3-4d52-9729-9e836f4c95e5.png)
![model_result](https://user-images.githubusercontent.com/109565405/180422384-55ab11cf-1617-4d97-8d03-9ef69ab34662.PNG)

- Data Preprocessing
X,features was scaled using MinMaxScaler and saved in pickle file.
y,target column was encoded with OneHotEncoder and saved in pickle file.
X, y splitted using train_test_plit. 

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

- Model Evaluation

![model_result](https://user-images.githubusercontent.com/109565405/180420180-7b4e9564-da45-4443-896e-a6f72f174aa1.PNG)
![confusion_matrix_display](https://user-images.githubusercontent.com/109565405/180420221-9c350209-ca49-477c-b234-24a3e4b7155a.png)

The results of this analysis: The accuracy obtained for this classification model is around 0.9 where about 8285 clients is correctly predicted tends to not deposit
money into banks based on some factors such as demographic profile and etc.


## Credits
Thanks to [HackerEarth HackLive: Customer Segmentation | Kaggle](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon). for contributions of customer segmentation dataset and my lecturer, Dr Warren for his sharing


