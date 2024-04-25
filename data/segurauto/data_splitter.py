import argparse
import random
import os

def split_data(file_list, train_percent, val_percent, test_percent):
    """Splits a list of file paths into train, val, and test sets.

    Args:
        file_list (list): List of file paths (each line from your train.txt)
        train_percent (float): Percentage of data for the training set.
        val_percent (float): Percentage of data for the validation set.
        test_percent (float): Percentage of data for the test set.
    """

    random.shuffle(file_list)  # Shuffle for random distribution

    total_count = len(file_list)
    train_count = int(total_count * train_percent)
    val_count = int(total_count * val_percent)

    train_set = file_list[:train_count]
    val_set = file_list[train_count:train_count + val_count]
    test_set = file_list[train_count + val_count:]

    return train_set, val_set, test_set


def write_to_file(data_set, file_name):
    """Writes a list of file paths to a text file.

    Args:
        data_set (list): List of file paths.
        file_name (str): Name of the file to write to.
    """
    with open(file_name, 'w') as f:
        for line in data_set:
            f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train, val, and test sets.")
    parser.add_argument("train_file", help="Path to the train.txt file.")
    parser.add_argument("train_percent", type=float, help="Percentage of data for training (0-1).")
    parser.add_argument("val_percent", type=float, help="Percentage of data for validation (0-1).")
    parser.add_argument("test_percent", type=float, help="Percentage of data for testing (0-1).")
    args = parser.parse_args()

    # Validate percentages
    if sum([args.train_percent, args.val_percent, args.test_percent]) != 1.0:
        raise ValueError("Percentages must add up to 1.0")

    with open(args.train_file, 'r') as f:
        lines = f.readlines()

    train_set, val_set, test_set = split_data(lines, args.train_percent, args.val_percent, args.test_percent)

    write_to_file(train_set, "train.txt")
    write_to_file(val_set, "val.txt")
    write_to_file(test_set, "test.txt")
