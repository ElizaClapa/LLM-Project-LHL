# <p style="text-align: center;"> ⚙️🔍🗣️ Large Language Model Project LHL 🗣️🔎🛠️</p>

##  Project Task 🕵️‍♂️🔎🔤
The Natural Language Processing (NLP) used is Sentiment Analysis. 

**Demonstration video ▶️:**
[![Sentiment video](https://github.com/ElizaClapa/LLM-Project-LHL/blob/main/Images%20-%20Videos/3%20Stars.png?raw=true)](https://www.youtube.com/watch?v=xvQY_AMO1y0)

## Dataset 📊📋📉
### 1. Dataset Summary

The *YelpReviewFull* dataset consists of reviews from Yelp. It was constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the Yelp Dataset Challenge 2015. 

It was first used as a text classification benchmark in the following paper:

Xiang Zhang, Junbo Zhao, Yann LeCun. *Character-level Convolutional Networks for Text Classification*. *Advances in Neural Information Processing Systems 28* (NIPS 2015).

### 2. Supported Tasks and Leaderboards

The dataset is mainly used for text classification: given the text, predict the sentiment.

### 3. Languages

The reviews were mainly written in English.


### 4. Data Instances

A typical data point, comprises of a text and the corresponding label. An example from the *YelpReviewFull* test set looks as follows:

```
{
    "label": 0,
    "text": 'I got \'new\' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the previous patch had blown - WAIT, WHAT? I just got the tire and never needed to have it patched? This was supposed to be a new tire. \\nI took the tire over to Flynn\'s and they told me that someone punctured my tire, then tried to patch it. So there are resentful tire slashers? I find that very unlikely. After arguing with the guy and telling him that his logic was far fetched he said he\'d give me a new tire \\"this time\\". \\nI will never go back to Flynn\'s b/c of the way this guy treated me and the simple fact that they gave me a used tire!'
}
```

### 5. Data Fields

```'text'```: The review texts are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".

```'label'```: Corresponds to the score associated with the review (between 1 and 5).

### 6. Data Splits

In total there are 650,000 trainig samples and 50,000 testing samples.

```
DatasetDict({
    train: Dataset({
        features: ['label', 'text'],
        num_rows: 650000
    })
    test: Dataset({
        features: ['label', 'text'],
        num_rows: 50000
    })
})
```
## Pre-trained Model 💻🈴🗃️
The pre-trained model used for the Sentiment Analysis task was [juliensimon/reviews-sentiment-analysis](juliensimon/reviews-sentiment-analysis).

This model was selected because it was fine-tuned using Distilbert model on English language reviews, and it was one of the most popular models available, with almost 1,000 downloads in the last month. 

## Performance Metrics 👎📏👍 
The performance metrics of the Optimized model were Accuracy, Precission, Recall, and F1-Score.

```
Evaluation Results:
{'eval_loss': 0.773500382900238, 'eval_accuracy': 0.684, 'eval_f1': 0.6833543859772582, 'eval_runtime': 98.6782, 'eval_samples_per_second': 5.067, 'eval_steps_per_second': 0.638}
Classification Report:
              precision    recall  f1-score   support

      1 star       0.79      0.78      0.79       110
      2 star       0.64      0.69      0.66       112
     3 stars       0.70      0.67      0.69        92
     4 stars       0.62      0.56      0.59       100
     5 stars       0.66      0.71      0.68        86

    accuracy                           0.68       500
   macro avg       0.68      0.68      0.68       500
weighted avg       0.68      0.68      0.68       500
```


| <p style="text-align: center;">text</p> | <p style="text-align: center;">label</p>|<p style="text-align: center;">score</p>|
|:----------|:----------|:----------|
| This restaurant was the best ever, I really enjoyed the food there!   | 5 stars    | 0.967317    |
| I would recommend this to my family and friends!    | 4 stars     | 0.530670    |
| Not that big of a deal, I don't know what everyone is talking about.   | 3 stars     | 0.626009   |
| It was okay, not that bad, but also not extremely good   | 3 stars    | 0.492008    |
| This was the worst meal I've ever had!   | 1 star    | 0.990348   |


## Hyperparameters 🙌🦾🦿💅💇‍♀️
Explain which hyperparameters you find most important/ relevant while optimizing your model.

The Hyperparameters changed for the optimization of this model include the following: 

```
training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5, 
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,  
    num_train_epochs=3,
    weight_decay=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='/content/drive/My Drive/Colab Notebooks/LLM Project GoogleColab/Logs_Full',
    logging_steps=10,
    push_to_hub=True,
    report_to="none"
)

optimized_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['test'],
    #train_dataset=small_train_dataset,
    #eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.001)]
)
```

The most important hyperparameter for this optimization was the **Learning Rate**, which had been modified from 5e-1 to 1e-1, and finally set to 2e-5 for the final optimization. A learning rate that's too high can cause the model to converge too quickly to a suboptimal solution, while too low a learning rate can result in slow convergence, resulting in long training times. A compromised between risk of suboptimal performance and training time was found with the final learning rate used (2e-5). 

Another hyperparameter changed was the number of training **Epochs**, which controls how many times the model sees the entire training dataset. Too few epochs may lead to underfitting, while too many can lead to overfitting. To avoid overfitting, a technique called **Early Stopping** was used. This technique is used to halt training when the model's performance on the validation set stops improving. This helps prevent overfitting by ensuring that the model does not continue training beyond the point where it is making significant progress.

Another important consideration was the **Weight Decay** hyperparameter, is it is useful for regularization to avoid overfitting. 

### Hyperparameters important for memory usage and speed

The following hyperparameters helped to avoid losing valuable model training progress due to the Colab Notebook disconecting from the hosted runtime due to inactivity or reaching the maximum RAM available: 
   
1. The **Per Device Evaluation Batch Size** directly affected the speed and memory usage during the evaluation.

2. The **Evaluation Strategy** was set to 'epoch' so the model would be evaluated on the validation set everytime one epoch was completed. 

3. The **Save Strategy** was set to 'epoch' so the models state would be saved with every completed epoch. 

Even if the notebook would disconnect, with the saved model's progress, the training could be restarted from that point. 

## Relevant Links 🔗🔗🔗

### Project Files Access 📂🗄️🗃️

Due to the large sizes of the project's files and limited bandwidth and Storage usage, the project had to be uploaded using Google Drive and providing the link to access it.

The provided link below includes the **README.md** file, the **'Notebooks and Data'** folder and the **'Images - Videos'** folder. 

#### **The data needed to run the notebooks is inside the *'Notebook and Data'* folder. They are in their respective directories relative to the notebooks' directory, which need to be saved inside the *'Notebooks and Data'* folder to run smoothly.** 

The 'Notebooks and Data' folder includes:

* 5 separate notebooks 📓📓📓 

* 1 'Models' folder 📂

* 1 'Logs_Full_Set' folder 📂

* 1 'Logs' folder 📂

* 1 'Logs_Full' folder 📂

* 1 'Models Results' folder 📂

* 1 'Data' folder 📂

### Google Drive Access Link 🔗
Access to the project's data files through this link:

https://drive.google.com/drive/folders/1ejjA_zWabYOxzgsKVvoGKYRaj5H93FmS?usp=share_link

### Link and screenshot of model on HuggingFace: 

Hugging Face Model Name and Link: 

[ElizaClaPa/SentimentAnalysis-YelpReviews-OptimizedModel🔗](https://huggingface.co/ElizaClaPa/SentimentAnalysis-YelpReviews-OptimizedModel).

![Model Card](https://github.com/ElizaClapa/LLM-Project-LHL/blob/main/Images%20-%20Videos/Hugging%20Face%20Model%20Card.png?raw=true)


### Link to your chosen dataset: 

The dataset can be found in Hugging Face, [YelpReviewFull🔗](https://huggingface.co/datasets/Yelp/yelp_review_full).

## Conclusion 

The Sentiment analysis model developed in this project can provide significant value to businesses in various ways. The following examples of practical applications and the value offered to businesses are briefly explained in the table below:


| Areas -> | Customer Feedback Analysis  | Brand Monitoring  | Marketing Campaign Evaluation  | Customer Service Improvement  | Product Reviews and Ratings Analysis  | Churn Prediction  |Personalized Marketing |
|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|
| Applications    | Automatically analyze customer reviews, social media mentions, and survey responses to predict the star ratings. | Monitor the predicted star ratings of reviews across various platforms to gauge overall brand sentiment. | Evaluate the impact of marketing campaigns by analyzing predicted star ratings before, during, and after the campaign.   |Analyze predicted star ratings from customer service interactions, such as chat logs, emails, and call transcripts. | Aggregate and analyze predicted star ratings from product reviews on e-commerce platforms.   | Analyze customer sentiment to predict potential churn by focusing on predicted low star ratings.    | Use predicted star ratings to tailor marketing messages to individual customers based on their expressed sentiments. |
| Value    | Improved customer experience (pain points and areas of improvement identified), Product Development based on customer preferences and feedback.    | Reputation Management (early detection of negative sentiment), and Competitor Analysis (identify their strengths and weaknesses)    | Campaign Optimization (identify aspects from campaigns that resonates with customers to improve future ones)    | Evaluate and improve customer service agents' performance based on sentiment trends in their interactions, and identify recurring issues in customer service and implement targeted training or process improvements.    | Sales Strategy (based on positive or negative trends in product reviews) and Inventory Management (stocking and discontinuing products based on customer sentiment)    | Retention Strategies and Proactive Engagement (with customers showing signs of negative sentiment to resolve issues before they lead to churn).    | Customer Engagement and Targeted Offers (promotions and discounts to customers based on their sentiment and behavior).


 ### Example of Sentiment Analysis Model in Action 👊🥐🥖🍪

**Scenario:**

🍞 Bread & Butter Bakery 🧈 launches a new line of gluten-free products. They want to evaluate customer sentiment and adjust their strategy accordingly. 

**1. Collect Reviews:** 👍👎 The bakery collects reviews from various platforms and uses the sentiment analysis model to predict the star ratings.

**2. Analyze Feedback:** 🕵️‍♀️🔍 The model predicts that most reviews are 4 or 5 stars, indicating positive reception. However, a subset of reviews (1-2 stars) mentions issues with the texture of the gluten-free bread.

**3. Adjust Strategy:**

* **Product Improvement:** The bakery's team works on improving the texture of the gluten-free bread based on feedback.

* **Customer Engagement:** They reach out to customers who left negative reviews, offering them a discount on their next purchase and informing them about the improvements made.

**4. Evaluate Campaign:** 📋📏The bakery runs a marketing campaign highlighting the improved gluten-free products. Post-campaign, they see an increase in 4-5 star reviews and a decrease in complaints about texture.

**5. Monitor Trends:** 📈📉The bakery continuously monitors review star ratings to ensure the new product maintains high customer satisfaction.

By leveraging the sentiment analysis model to predict star ratings, 🍞 Bread & Butter Bakery 🧈 can make data-driven decisions to enhance their product offerings, improve customer service, and optimize marketing strategies, ultimately leading to increased customer satisfaction and business growth.