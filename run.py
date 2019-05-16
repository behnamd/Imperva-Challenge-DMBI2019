import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def main():
    input_directory = "Data/"
    input_file = "train_data.csv"
    full_path = input_directory + input_file

    # read targeted file
    df = pd.read_csv(full_path)

    df = df.fillna(-1)
    df.replace(False, 0, inplace=True)
    df.replace(True, 1, inplace=True)

    # split to train and test
    training_percent = 0.8
    (x_train, y_train), (x_test, y_test) = split_training_and_test(training_percent, df)

    x_train.to_csv(input_directory + "training.csv")
    x_test.to_csv(input_directory + "test.csv")

    # default_classifier = RandomForestClassifier()
    # default_classifier.fit(x_train, y_train)
    # important_features_df = self._get_feature_importance_greater_than_zero(default_classifier, X_original_train)
    # important_features_df.to_csv(self._output_path + "Training_Feature_Importance_No_Validation_Random_Forest.csv")

    print("Done!!")



def load_data(path='train_data.csv',training_percent=0.8):
    df = pd.read_csv(path)
    df = df.fillna(-1)
    df.replace(False, 0, inplace=True)
    df.replace(True, 1, inplace=True)
    (x_train, y_train), (x_test, y_test) = split_training_and_test(training_percent, df)
    return (x_train.values, y_train.values), (x_test.values, y_test.values)




def split_training_and_test(training_percent, df):
    x_train = df.sample(frac=training_percent)
    x_test = df[~df.index.isin(x_train.index)]


    y_train = x_train.pop("user.id")
    y_test = x_test.pop("user.id")

    return (x_train, y_train), (x_test, y_test)

def calculate_f1(actual, predictions):
    f1 = f1_score(actual, predictions)
    return f1


def get_feature_importance_greater_than_zero(classifier, X_train):
    features_df = pd.DataFrame()
    features_df["score"] = classifier.feature_importances_
    features_df["name"] = X_train.columns

    important_features_df = features_df[features_df["score"] > 0]
    important_features_df = important_features_df.sort_values(by='score', ascending=False)
    return important_features_df

if __name__ == '__main__':
    main()