"""
This file contains variables and markup text that require global scope.
"""

ASSETS = "./asset/"

# markup text
markup_text = {
    'homepage': """
                This is a simple web app that showcases the following:
                <ul>
                    <li> Churn Analysis on the customer <a href = 'https://www.kaggle.com/becksddf/churn-in-telecoms-dataset'>data</a>.
                    <li> Topic Modeling on <a href = 'www.google.com'>customer call transcript</a>.
                <ul>
                
                Churn Modules further provides 3 options as follows:
                <ul>
                    <li> <b>Exploratory Data Analysis</b> <br>
                            This sub-module shows different distributions and trends that <a href = 'https://www.kaggle.com/becksddf/churn-in-telecoms-dataset'>data</a> 
                            follows and presents them in a clean and interactive way.
                    <li> <b>Training models</b> <br>
                            This sub-module provides dual functionality to either train a model or simply load a pre-trained model
                            and check out different metrics like Accuracy, Precision and Recall. 
                    <li> <b>Expandability</b><br>
                            This sub-module tried to explain various different aspects of the model as a whole. Using SHAP 
                            and PDP plots we can estimate the importance that model gives to attributes.                     
                </ul>
                
                Topic Modeling module further provides options as follows:
                <ul>
                <li> <b>Classify</b> <br>
                Identify the latent (or underlying) topic of the document
                <li> <b>Explain</b> <br>
                Explain why a particular topic was picked up.
                </ul>
                
                Inference module further provides options as follows:
                <ul>
                <li> A clean way to visualize results from Churn Analysis and Topic modules
                </ul>
                <hr>
                Note: Further instructions are also available in modules/sub-modules.
                """
}

# churn global variables
churn_dict = {
    'demo_key': 'demo_value'
}

# intent global variables
intent_dict = {
    'demo_key': 'demo_value'
}

# model_paths
model_path = {
    'churn_model': './data/models/',
    'intent_model': './data/models'
}
