import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib



import gender_guesser.detector as gender



from sklearn.model_selection import StratifiedKFold, train_test_split,RandomizedSearchCV

from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import learning_curve

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier


def read_user_datasets():
    # Reads users profile from csv files
    real_users = pd.read_csv(r"C:\Users\vardh\OneDrive\Desktop\SIH 2023\fusers.csv")
    fake_users = pd.read_csv(r"C:\Users\vardh\OneDrive\Desktop\SIH 2023\users.csv")

    x = pd.concat([real_users, fake_users])
    y = len(real_users) * [0] + len(fake_users) * [1]
    print(y)

    return x,y


def predict_user_sex(name):
    d = gender.Detector(case_sensitive=False)
    first_name = str(name).split(' ')[0]
    gen = d.get_gender(u"{}".format(first_name))

    gender_code_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'andy': 0, 'mostly_male': 1, 'male': 2}
    code = gender_code_dict[gen]

    return code


def extract_user_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))

    lang_dict = {name: i for i, name in lang_list}

    x.loc[:, 'lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)
    x.loc[:, 'gender_code'] = predict_user_sex(x['name'])

    feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
                              'gender_code', 'lang_code']
    x = x.loc[:, feature_columns_to_use]

    return x


x,y = read_user_datasets()
print("dataset read complete.....")

#extract features
x = extract_user_features(x)
print(x.columns)
print(x.head(5))

#spliting
XX_train,XX_test,yy_train,yy_test = train_test_split(x, y, test_size=0.20, random_state=100,shuffle=True)

#Random forest object
rf_classifier = RandomForestClassifier()

#hyperparameter tuning
n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)]

parameters = {
    'n_estimators': n_estimators,
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 10, 15, 30, 50],
    'max_features': ['sqrt', 'log2', None],
    'random_state': [42],
    'bootstrap': [True, False]
}

clf = RandomizedSearchCV(estimator=rf_classifier, param_distributions=parameters, cv=10, verbose=2, n_jobs=4)
clf.fit(XX_train, yy_train)


#rf_classifier.fit(XX_train, yy_train)


#saving the model
joblib.dump(clf, 'fake_pro_model.pkl')

train_predictions = clf.predict(XX_train)
prediction = clf.predict(XX_test)

err_training = mean_absolute_error(train_predictions, yy_train)
err_test = mean_absolute_error(prediction, yy_test)

#plot_roc_curve(yy_test, prediction)

print("Train Accuracy is : {}".format(100 - (100*err_training)))
print("Test Accuracy is : {}".format(100 - (100*err_test)))

print(f'Train accuracy: {clf.score(XX_train, yy_train):.3f}')
print(f'Test accuracy: {clf.score(XX_test, yy_test):.3f}')

#confusionMatrixx = confusion_matrix(yy_test, prediction)
#print('Confusion matrix, without normalization')
#print(confusionMatrixx)
#plot_confusion_matrix(confusionMatrixx)



#input
def get_user_input():
    name = input("Enter the name: ")
    statuses_count = int(input("Enter statuses count: "))
    followers_count = int(input("Enter followers count: "))
    friends_count = int(input("Enter friends count: "))
    favourites_count = int(input("Enter favourites count: "))
    listed_count = int(input("Enter listed count: "))
    lang = input("Enter language: ")

    user_data = {
        'name': name,
        'statuses_count': statuses_count,
        'followers_count': followers_count,
        'friends_count': friends_count,
        'favourites_count': favourites_count,
        'listed_count': listed_count,
        'lang': lang
    }

    return pd.DataFrame([user_data])


def predict_profile_authenticity(user_data):
    # Extract features for the given user data
    user_features = extract_user_features(user_data)

    # Predict whether the profile is fake or real
    prediction = clf.predict(user_features)
    return prediction[0]

while True:
    # Get user input for a profile
    user_data = get_user_input()

    # Predict the authenticity of the profile
    prediction = predict_profile_authenticity(user_data)

    if prediction == 0:
        print("The profile is predicted to be real.")
    else:
        print("The profile is predicted to be fake.")

    cont = input("Do you want to check another profile? (yes/no): ")
    if cont.lower() != 'yes':
        break