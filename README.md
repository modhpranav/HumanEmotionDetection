
# Human Emotion Detection

Facial Emotion Detection is a project which uses facial images to train Logistic regression and CNN models.
After model training we made inference on the model.
[Repository Link](https://github.com/pranav-modh/HumanEmotionDetection)


## Installation

- Clone the repository using git clone.
- Install requirements from the requirements.txt

```bash
  cd AML_Project
  pip3 install -r requirements.txt
```
## Model Inference


- Now for inferencing model make sure models are downloaded into the local system.
    - For Linux/MacOS:
    ```bash
    ls models/CNN_Models
    --> Terminal Output:
        CNN_8_layer_model_50_epoch_final.h5
    ```

    - For Windows:
    ```
    dir /b models\CNN_Models
    --> Terminal Output:
        CNN_8_layer_model_50_epoch_final.h5
    ```
- If Outputs are as shown above skip this step.

   -> To download models manually if not getting downloaded automatically using git.\
    -> [Visit this folder](https://drive.google.com/drive/folders/1oME7S3Q_coVU7cllfY-AKIz-Jw_6WEX2?usp=share_link) and download ```.h5``` file from the folder.

- Now execute ```inference.py``` file using below command.
    ```
    python/python3 inference.py
    ```
- Follow the instructions displayed in the terminal after execution is completed.


## Dataset Download

- To download full dataset for training model use below folders.
    - [Training And Validation Dataset For CNN Model](https://drive.google.com/drive/folders/1dWaGT6ExRPUtkTzAZCo8PgR1GLvIwhYA?usp=share_link)
    - [Training And Test Dataset For Logistic Regression Model](https://drive.google.com/drive/folders/1fbDE13WLFFfOE2QQJLmgHfPJ9pqwctL1?usp=share_link)

## Dataset Download

- To download full dataset for training model use below folders.
    - [Training And Validation Dataset For CNN Model](https://drive.google.com/drive/folders/1dWaGT6ExRPUtkTzAZCo8PgR1GLvIwhYA?usp=share_link)
    - [Training And Test Dataset For Logistic Regression Model](https://drive.google.com/drive/folders/1fbDE13WLFFfOE2QQJLmgHfPJ9pqwctL1?usp=share_link)

## CNN Model Training

- When you cloned repository a notebook is also downloaded by name ```AML_Project_CNN.ipynb```
- Open notebook in jupyter notebook or colab.
- Follow each header to install requirements and then train the model.
- __Make sure to pass dataset folder path of your own system__

## Logistic Regression Model Training 

- When you cloned repository a notebook is also downloaded by name ```AML_Project_LogisticRegression.ipynb```
- Open notebook in jupyter notebook or colab.
- Follow each header to install requirements and then train the model.
- __Make sure to pass dataset folder path of your own system__
## Project Report

- [Report](https://docs.google.com/document/d/e/2PACX-1vQNZPkuu6HLuurxxyYM5v8EMx9SDlibkFBwZtEtShGMxO-xG40KedaBVduutdPs4WsL9i4lJRLNWCkq/pub)
## Acknowledgements

 - [Google Image Search](https://pypi.org/project/Google-Images-Search/)
 - [Pyppeteer](https://github.com/pyppeteer/pyppeteer)
 - [Prof. Ozgun Babur - CS638](https://www.umb.edu/ozgun_babur)




## Authors

- [@modhpranav](https://www.github.com/modhpranav)


## License

[MIT](https://choosealicense.com/licenses/mit/)

