# AI Workflow: AI In Production

* **cs-train**: Contains all the data to train the model
* **models**: Contains all pre-trained saved models for prediction
* **notebooks**: Contains all the notebooks describing solutions and depicting visualizations
* **templates**: Simple templates for rendering flask app
* **unittest**: It has logger test, API test and model test for testing all the functionalities before deploying to production and for maintenance post deployment
* **Dockerfile**: Contains all the commands a user could call on the command line to assemble the docker image.
* **app.py**: Flask app for creating a user interface /train and /predict APIs in order to train and predict respectively
* **cslib.py**: A collection of functions that will transform the data set into features you can use to train a model.
* **model.py**:  A module having functions for training, loading a model and making predictions
