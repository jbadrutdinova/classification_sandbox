import pandas as pd

from sklearn import metrics
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier


def optimal_max_depth_dt(values, criterion, random_state, x_train, y_train, x_test, y_test):

    accuracy_max_depth = []
    for val in values:
        DTs = DecisionTreeClassifier(criterion = criterion,
                                    max_depth = val,
                                    random_state = random_state)
        DTs.fit(x_train, y_train)
        y_pred = DTs.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy_max_depth.append(accuracy)

    max_depth_df = pd.DataFrame({
        'max_values': values,
        'Accuracy': accuracy_max_depth
    })

    highest_acc_max_depth = max_depth_df.loc[max_depth_df['Accuracy'].idxmax()]
    md = int(highest_acc_max_depth['max_values'])

    print(f'Max Depth value with highest model accuracy: {md}')
    display(highest_acc_max_depth)
    
    return md


def optimal_min_samples_leaf_dt(values, criterion, md, random_state, x_train, y_train, x_test, y_test):
    
    accuracy_min_samples_leaf = []
    for val in values:
        DTs = DecisionTreeClassifier(criterion = criterion,
                                     max_depth = md,
                                     random_state = random_state,
                                     min_samples_leaf = val)
        DTs.fit(x_train, y_train)
        y_pred = DTs.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy_min_samples_leaf.append(accuracy)

    min_samples_leaf_df = pd.DataFrame({
        'leaf_values': values,
        'Accuracy': accuracy_min_samples_leaf
    })

    highest_acc_min_samples_leaf = min_samples_leaf_df.loc[min_samples_leaf_df['Accuracy'].idxmax()]
    msl = int(highest_acc_min_samples_leaf['leaf_values'])

    print(f'Min Samples Leaf value with highest model accuracy: {msl}')
    display(highest_acc_min_samples_leaf)

    return msl


def optimal_min_samples_split_dt(values, criterion, md, msl, random_state, x_train, y_train, x_test, y_test):

    accuracy_min_samples_split = []
    for val in values:
        DTs = DecisionTreeClassifier(criterion = criterion,
                                    max_depth = md,
                                    random_state = random_state,
                                    min_samples_leaf = msl,
                                    min_samples_split = val)
        DTs.fit(x_train, y_train)
        y_pred = DTs.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy_min_samples_split.append(accuracy)

    min_samples_split_df = pd.DataFrame({
        'split_values': values,
        'Accuracy': accuracy_min_samples_split
    })

    highest_acc_min_samples_split = min_samples_split_df.loc[min_samples_split_df['Accuracy'].idxmax()]
    mss = int(highest_acc_min_samples_split['split_values'])

    print(f'Min Samples Split value with hichest model accuracy: {mss}')
    display(highest_acc_min_samples_split)

    return mss
