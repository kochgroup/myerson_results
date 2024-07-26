def random_split(dataset, splits):
    len_dataset = len(dataset)
    end_train = int(len_dataset * splits[0])
    end_val = len_dataset if len(splits) <= 2 else end_train+int(len_dataset*splits[1])
    dataset_train = dataset[:end_train]
    dataset_val = dataset[end_train:end_val]
    dataset_test = dataset[end_val:len_dataset]
    return [dataset_train, dataset_val, dataset_test]

split_dict = {
    'random': random_split
}

def dataset_split(key:str):
    return split_dict[key]