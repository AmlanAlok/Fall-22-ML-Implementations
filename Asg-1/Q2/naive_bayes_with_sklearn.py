from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):

    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


def read_data(filename, col_headers):

    dataset_path = '../dataset/'
    file_path = dataset_path + filename
    input_data = fetch_data(file_path)
    # df = pd.DataFrame(input_data, columns=col_headers)
    df = pd.DataFrame(input_data)
    return df


def naive_bayes_implementation(train_df, test_df):
    x_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    x_test = test_df

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = GaussianNB()
    clf.fit(x_train_scaled, y_train)

    test_predict = clf.predict(x_test_scaled)
    p = clf.predict_proba(x_test_scaled)
    return test_predict


def library_output():
    train_filename = '1a-training.txt'
    test_filename = '1a-test.txt'

    training_col_header = ['height', 'weight', 'age', 'col']
    test_col_header = ['height', 'weight', 'age']

    train_df = read_data(train_filename, training_col_header)
    test_df = read_data(test_filename, test_col_header)

    y_pred = naive_bayes_implementation(train_df, test_df)
    print(y_pred)
    return y_pred


if __name__ == '__main__':
    library_output()