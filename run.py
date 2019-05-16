import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def main():
    input_directory = "data/"
    input_file = "train_data.csv"
    full_path = input_directory + input_file
    df = pd.read_csv(full_path)

    training_percent = 0.8
    training_df, test_df = split_training_and_test(training_percent, df)

    training_df.to_csv(input_directory + "training.csv")
    test_df.to_csv(input_directory + "test.csv")

    print("Done!!")




def split_training_and_test(training_percent, df):
    training_df = df.sample(frac=training_percent)
    test_df = df[~df.index.isin(training_df.index)]

    return training_df, test_df

def calculate_f1(actual, predictions):
    f1 = f1_score(actual, predictions)
    return f1

if __name__ == '__main__':
    main()