"""
Internal file: To be used only for debugging purposes.
"""

from helper.explain import Explain

explain = Explain(model_name='Logistic Regression',
                  model_path='./data/model',
                  data_path='./data/output',
                  only_test=False)

explain.run()


"""
import sklearn
import shap
from sklearn.model_selection import train_test_split

# print the JS visualization code to the notebook
shap.initjs()

# train a SVM classifier
X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
svm = sklearn.svm.SVC(kernel='rbf', probability=True)
svm.fit(X_train, Y_train)

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

# plot the SHAP values for the Setosa output of the first instance

show_plot(
    shap.force_plot(explainer.expected_value[1], shap_values[1][:, :], X_test.iloc[:, :], link="logit", show=True),
    400, 1200)

st.pyplot(shap.summary_plot(shap_values, X_test))
"""
