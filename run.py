import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def main():
    input_directory = "data/"
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

    print("Done!!")




def split_training_and_test(training_percent, df):
    x_train = df.sample(frac=training_percent)
    x_test = df[~df.index.isin(x_train.index)]

    y_train = x_train["user.id"]
    y_test = x_test["user.id"]

    return (x_train, y_train), (x_test, y_test)

def calculate_f1(actual, predictions):
    f1 = f1_score(actual, predictions)
    return f1

if __name__ == '__main__':
    main()