import pandas as pd

from sklearn import metrics
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier


def optimal_max_depth_rf(values, random_state, x_train, y_train, x_test, y_test):

    accuracy_max_depth = []
    for val in values:
        RF2 = RandomForestClassifier(random_state = random_state,
                                     max_depth = val) 
        RF2.fit(x_train, y_train.ravel())
        y_pred = RF2.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy_max_depth.append(accuracy)

    max_depth_df = pd.DataFrame({
        'max_depth_values': values,
        'Accuracy': accuracy_max_depth
    })

    highest_acc_max_depth = max_depth_df.loc[max_depth_df['Accuracy'].idxmax()]
    md = int(highest_acc_max_depth['max_depth_values'])

    print(f'Max Depth value with highest model accuracy: {md}')
    display(highest_acc_max_depth)

    return md


def optimal_n_estimators_rf(values, random_state, md, x_train, y_train, x_test, y_test):

    accuracy_n_estimators = []
    for val in values:
        RF2 = RandomForestClassifier(random_state = random_state,
                                     max_depth = md,
                                     n_estimators = val) 
        RF2.fit(x_train, y_train.ravel())
        y_pred = RF2.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy_n_estimators.append(accuracy)

    n_estimators_df = pd.DataFrame({
        'n_estimators_values': values,
        'Accuracy': accuracy_n_estimators
    })

    highest_acc_n_estimators = n_estimators_df.loc[n_estimators_df['Accuracy'].idxmax()]
    n_e = int(highest_acc_n_estimators['n_estimators_values'])
    
    print(f'N Estimators value with highest model accuracy: {n_e}')
    display(highest_acc_n_estimators)

    return n_e


def optimal_min_samples_leaf_rf(values, random_state, md, n_e, x_train, y_train, x_test, y_test):

    accuracy_min_samples_leaf = []
    for val in values:
        RF2 = RandomForestClassifier(random_state = random_state,
                                     max_depth = md,
                                     n_estimators = n_e,
                                     min_samples_leaf = val) 
        RF2.fit(x_train, y_train.ravel())
        y_pred = RF2.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy_min_samples_leaf.append(accuracy)

    min_samples_leaf_df = pd.DataFrame({
        'min_samples_leaf_values': values,
        'Accuracy': accuracy_min_samples_leaf
    })

    highest_acc_min_samples_leaf = min_samples_leaf_df.loc[min_samples_leaf_df['Accuracy'].idxmax()]
    msl = int(highest_acc_min_samples_leaf['min_samples_leaf_values'])
    
    print(f'Mean Samples Leaf value with highest model accuracy: {msl}')
    display(highest_acc_min_samples_leaf)

    return msl


def optimal_min_samples_split_rf(values, random_state, md, n_e, x_train, y_train, x_test, y_test):

    accuracy_min_samples_split = []
    for val in values:
        RF2 = RandomForestClassifier(random_state = random_state,
                                     max_depth = md,
                                     n_estimators = n_e,
                                     min_samples_split = val) 
        RF2.fit(x_train, y_train.ravel())
        y_pred = RF2.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy_min_samples_split.append(accuracy)

    min_samples_split_df = pd.DataFrame({
        'split_values': values,
        'Accuracy': accuracy_min_samples_split
    })

    highest_acc_min_samples_split = min_samples_split_df.loc[min_samples_split_df['Accuracy'].idxmax()]
    mss = int(highest_acc_min_samples_split['split_values'])
    
    print(f'Mean Samples Split value with highest model accuracy: {mss}')
    display(highest_acc_min_samples_split)

    return mss
