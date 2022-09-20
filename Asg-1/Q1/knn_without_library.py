import inspect
import math

CONST_PREDICTION = 'prediction'


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):

    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()

    clean_input = change_data_structure(clean_input)
    return clean_input


def change_data_structure(input_data):

    data_list = [None] * len(input_data)
    i = 0

    for record in input_data:
        data_list[i] = {
            'index': i,
            'input': {
                'height': float(record[0]),
                'weight': float(record[1]),
                'age': int(record[2])
            },
            'output': record[3]
        }

        i += 1

    print('Converted each data point to dictionary format')

    return data_list


def get_distance(input_data, test_input):

    a = [None]*len(input_data)
    i = 0

    for dp in input_data:
        cartesian_distance = math.sqrt(
            pow(dp['input']['height']-test_input['input']['height'], 2) +
            pow(dp['input']['weight']-test_input['input']['weight'],2) +
            pow(dp['input']['age']-test_input['input']['age'], 2))

        a[i] = {
            'index': dp['index'],
            'cartesian_distance': cartesian_distance,
            'output': dp['output']
        }

        i += 1

        # print('sqrt((', dp['input']['height'] ,'-', test_input['input']['height'], ')^2 +'+
        #     '(', dp['input']['weight'] ,'-', test_input['input']['weight'], ')^2 +'+
        #     '(', dp['input']['age'] ,'-', test_input['input']['age'], ')^2) =', cartesian_distance)

    sorted_a = sorted(a, key=lambda d: d['cartesian_distance'])

    print(sorted_a)

    return sorted_a


def make_prediction(k_list, distance_array, test_input_record):

    print('inside func ' + inspect.stack()[0][3])
    print('------------------------------------------------')
    for k in k_list:

        print('Executing for k =', k)
        w_count = 0
        m_count = 0

        knn_array = distance_array[:k]

        for dic in knn_array:
            if dic['output'] == 'W':
                w_count += 1
            if dic['output'] == 'M':
                m_count += 1

        W_prob = w_count / k
        M_prob = m_count / k

        print('W prob =', W_prob)
        print('M prob =', M_prob)

        k_dict = {
            'k': k,
            'prediction': ''
        }

        if w_count > m_count:
            k_dict[CONST_PREDICTION] = 'W'
        elif m_count > w_count:
            k_dict[CONST_PREDICTION] = 'M'
        else:
            k_dict[CONST_PREDICTION] = '50-50 M/W'

        test_input_record['output'].append(k_dict)
        print('prediction =', k_dict)
        print('------------------------------------------------')

    return test_input_record


def scratch_code_output():
    train_filename = '1a-training.txt'
    test_filename = '1a-test.txt'

    dataset_path = '../dataset/'
    input_data = fetch_data(dataset_path + train_filename)
    user_input = '( 1.7512428413306, 73.58553700624, 34)'
    user_input = clean_data(user_input)

    k_list = [1, 3, 5]

    test_input_record = {
        'input': {
            'height': float(user_input[0]),
            'weight': float(user_input[1]),
            'age': int(user_input[2])
        },
        'knn': k_list,
        'output': []
    }

    '''This gives the cartesian distance from the test dp'''
    distance_array = get_distance(input_data, test_input_record)

    test_input_record = make_prediction(k_list, distance_array, test_input_record)

    print('END')


if __name__ == '__main__':
    scratch_code_output()
