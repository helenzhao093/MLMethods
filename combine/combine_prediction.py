import numpy as np

def describe(predictions):
    num_classifiers = len(predictions)
    num_examples = len(predictions[0])
    num_classes = len(predictions[0][0])
    return num_classifiers, num_examples, num_classes

def majority_vote(labels):
    num_examples = len(labels[0])
    num_classifiers = len(labels)
    
    final_prediction = []
    for example_index in range(num_examples):
        counts = {}
        for clf_index in range(num_classifiers):
            if labels[clf_index][example_index] not in counts:
                counts[labels[clf_index][example_index]] = 0
            counts[labels[clf_index][example_index]] += 1
        
        final_prediction.append(max(counts, key=counts.get))
    return np.array(final_prediction)

def product_rule(predictions):
    num_classifiers, num_examples, num_classes = describe(predictions)
    final_prediction = []
    final_proba = []
    for example_index in range(num_examples):
        products = []
        for class_index in range(num_classes):
            product = 1.0
            for clf_index in range(num_classifiers):
                product = product * predictions[clf_index][example_index][class_index]
            products.append(product)
        # get index of max 
        index_max = np.array(products).argmax(axis=0)
        final_prediction.append(index_max)
        final_proba.append(products)
    return np.array(final_prediction), np.array(final_proba)

def max_rule(predictions):
    num_classifiers, num_examples, num_classes = describe(predictions)
    final_prediction = []
    final_proba = []
    for example_index in range(num_examples):
        maxes = []
        for class_index in range(num_classes):
            current_max = 0.0
            for clf_index in range(num_classifiers):
                current_max = max(predictions[clf_index][example_index][class_index], current_max)
            maxes.append(current_max)
        # get index of max 
        index_max = np.array(maxes).argmax(axis=0)
        final_prediction.append(index_max)
        final_proba.append(maxes)
    return np.array(final_prediction), np.array(final_proba)

def median_rule(predictions):
    num_classifiers, num_examples, num_classes = describe(predictions)
    final_prediction = []
    final_proba = []
    for example_index in range(num_examples):
        sums = []
        for class_index in range(num_classes):
            current_sum = 0.0
            for clf_index in range(num_classifiers):
                current_sum = current_sum + predictions[clf_index][example_index][class_index]
            sums.append(current_sum / num_classifiers)
        # get index of max 
        index_max = np.array(sums).argmax(axis=0)
        final_prediction.append(index_max)
        final_proba.append(sums)
    return np.array(final_prediction), np.array(final_proba)

def sum_rule(predictions):
    num_classifiers, num_examples, num_classes = describe(predictions)
    final_prediction = []
    final_proba = []
    for example_index in range(num_examples):
        sums = []
        for class_index in range(num_classes):
            current_sum = 0
            for clf_index in range(num_classifiers):
                current_sum = current_sum + predictions[clf_index][example_index][class_index]
            sums.append(current_sum)
        # get index of max 
        index_max = np.array(sums).argmax(axis=0)
        final_prediction.append(index_max)
        final_proba.append(sums)
    return np.array(final_prediction), np.array(final_proba)

def min_rule(predictions):
    num_classifiers, num_examples, num_classes = describe(predictions)
    final_prediction = []
    final_proba = []
    for example_index in range(num_examples):
        mins = []
        for class_index in range(num_classes):
            current_min = 5
            for clf_index in range(num_classifiers):
                current_min = min(predictions[clf_index][example_index][class_index], current_min)
            mins.append(current_min), np.array(final_proba)
        # get index of max 
        index_max = np.array(mins).argmax(axis=0)
        final_prediction.append(index_max)
        final_proba.append(mins)
    return np.array(final_prediction), np.array(final_proba)