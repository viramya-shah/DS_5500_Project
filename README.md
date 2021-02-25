# Improving Customer Experience
## Churn Analysis

_Juhi Paliwal - Viramya Shah_

### Motivation
<center>
_“It is better to stay focused on customers as they are the ones paying for your services. Competitors are never going to give you any money.” 
- Jeff Bezos
</center>

<hr>
The most valuable asset any organization can ever have is the clientele and the data regarding their behavior. For such companies, losing even a single customer is analogous to losing an edge over the competition. Such tendency highly correlates with lost revenue and increased acquisition spend i.e. it plays a very nuanced role in a company’s growth potential. Thus, it is of utmost importance to look after the customer needs and their interests. More formally, this is known as customer churn analysis. This kind of analysis yields results that help the organization identify behavior patterns of potential leaving customers, group these at-risk customers and rectify their action plan to gain the trust back.
Extending data science knowledge to this problem can be helpful to a great extent. Primarily, we can identify the specific customers that might churn in the following time frame. Diving deeper into the problem, information from data models can also help companies identify which attributes/services are a deal-breaker for the customers and thus can be improved upon. Specifically, attributes like tenure, average customer spending, customer location and age say a lot about the possible customer behavior. Above stated problem is exactly where this group is aligning to and we are focusing on a telecom-based customer service dataset[1]. Since this is a very important problem for an organization to thrive on, much work has been done. But the core problem is that these models are biased to certain customers, fine-tuned to individual missions, too complex for laymen to understand, or all three. Thus, we envision our phase one’s outcome to be a simple web application that hosts an unbiased, and explainable machine learning model that accurately predicts whether the customer would discontinue the company’s offering or not.

### Methodology
* Low Risk: The low risk problem that we have identified revolves around the data consistency. Even a medium-sized company’s customer service data could be in GBs, and thus there is a risk of data being non-homogenous and having technical noise. Thus, the first step would be to clean out the data by removing these kinds of noisy data points. We hypothesize that outliers would be a very small fraction of the data and removing them would not result in losing information. We plan to employ clustering methods (with n_clusters = 1) on the data (we will ignore the target variable for now) and calculating Silhouette Coefficient and Calinski and Harabasz scores[2]. Lower scores would imply that data is equally distributed and doesn’t have.

* Medium Risk: The medium risk problem for the given dataset would be the class imbalance. Lower churn scores are always beneficial to any company but that also means there would be a significant amount of class imbalance in the dataset. Such kind of imbalance either negatively impacts the model performance, or gives an overestimate of the metrics like Accuracy. To handle this risk, we plan to use the imblearn[3] package to tackle class imbalance with methods like SMOTE or over/under sampling. Logistic Regression, Support Vectors, and Forest models would follow after the imbalance problem.

* High Risk: One of the main motivations for us is to make the model as transparent or as simple as possible for non-technical people to understand. In other words, we want to be able to justify the model outcome. For example, if the model predicts that a particular customer might discontinue then we want to also know what particular attributes of that customer resulted in the model prediction. Was average spending an issue? Was the location an issue? Unlike feature selection which is used before modeling, this method comes after the modeling part and makes the model “explainable”. To handle this risk, we plan to use IBM's AIX360[4] package. Using PDPs, SHAP and pertinents, we would try to explain the model.

### Impact
With all the risks described above, we are envisioning this web application to provide a simple, unified 1-stop solution for customer service centers. This kind of application wouldn’t not only help the customer service employees change their approach in real time, but also provide more insights to deal with the customer. Besides it would also help the company leadership understand which attributes are negatively impacting their growth and what business actions need to be taken to either mitigate or eliminate them altogether.

### Reference

[1] https://www.kaggle.com/becksddf/churn-in-telecoms-dataset

[2] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

[3] https://github.com/scikit-learn-contrib/imbalanced-learn

[4] https://github.com/Trusted-AI/AIX360 
