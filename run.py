import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import scipy.stats

def main():
    input_directory = "Data/"
    input_file = "train_data.csv"
    full_path = input_directory + input_file

    print("Read CSV file")
    # read targeted file
    df = pd.read_csv(full_path)

    print("Remove Nan values")
    df = df.fillna(-1)
    df.replace(False, 0, inplace=True)
    df.replace(True, 1, inplace=True)

    # print("Split to train and test")
    # # split to train and test
    # training_percent = 0.8
    # (x_train, y_train), (x_test, y_test) = split_training_and_test(training_percent, df)
    #
    # x_train.to_csv(input_directory + "training.csv")
    # x_test.to_csv(input_directory + "test.csv")


    x_train = df
    x_train = remove_lower_entropy(x_train, input_directory)
    x_train.to_csv(input_directory + "higher_entropy_training.csv")

    print("Number of entropy higher than 0: {0}", len(list(x_train.columns)))


    # default_classifier = RandomForestClassifier()
    # default_classifier.fit(x_train, y_train)
    # important_features_df = self._get_feature_importance_greater_than_zero(default_classifier, X_original_train)
    # important_features_df.to_csv(self._output_path + "Training_Feature_Importance_No_Validation_Random_Forest.csv")

    print("Done!!")



def ent(data):
    """Calculates entropy of the passed `pd.Series`
    """
    p_data = data.value_counts()           # counts occurrence of each value
    entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    return entropy


def remove_lower_entropy(x_train, input_directory):
    cols = list(x_train.columns)
    entropies = []
    for column in cols:
        series = x_train[column]
        entropy = ent(series)
        column_entropy = (column, entropy)
        entropies.append(column_entropy)
    entropy_df = pd.DataFrame(entropies, columns=['Feature', 'Entropy'])
    sorted_entropy_df = entropy_df.sort_values(by=['Entropy'], ascending=False)

    higher_entropy_df = sorted_entropy_df[sorted_entropy_df['Entropy'] > 0]
    higher_entropy_df.to_csv(input_directory + "higher_entropy_df.csv")

    zero_entropy_df = sorted_entropy_df[sorted_entropy_df['Entropy'] == 0]

    zero_entropy_feature_series = zero_entropy_df['Feature']
    zero_entropy_feature_names = zero_entropy_feature_series.tolist()
    x_train = x_train.drop(zero_entropy_feature_names, axis=1)
    return x_train


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