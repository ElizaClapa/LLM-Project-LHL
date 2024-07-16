# <p style="text-align: center;"> ⚙️🔍🗣️ Large Language Model Project LHL 🗣️🔎🛠️</p>

##  Project Task 🕵️‍♂️🔎🔤
The Natural Language Processing (NLP) used is Sentiment Analysis. 

**Demonstration video ▶️ :**
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


### Link and screenshot of model on HuggingFace: 

Hugging Face Model Name and Link: [ElizaClaPa/SentimentAnalysis-YelpReviews-OptimizedModel🔗](https://huggingface.co/ElizaClaPa/SentimentAnalysis-YelpReviews-OptimizedModel).

![Model Card](https://github.com/ElizaClapa/LLM-Project-LHL/blob/main/Images%20-%20Videos/Hugging%20Face%20Model%20Card.png?raw=true)


### Link to your chosen dataset: 

The dataset can be found in Hugging Face, [YelpReviewFull🔗](https://huggingface.co/datasets/Yelp/yelp_review_full).
