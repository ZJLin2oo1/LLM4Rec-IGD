import os
import csv
import json
import argparse
from collections import Counter
from typing import Dict
# change1

def load_item_titles(item_meta_file: str) -> Dict[int, str]:
    """
    Load the mapping from item_id to item_title.

    Args:
        item_meta_file (str): Path to the file containing item titles and IDs.

    Returns:
        Dict[int, str]: Mapping from item_id to item_title.
    """
    item_dict = {}

    with open(item_meta_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            item_title = '\t'.join(parts[:-1])
            item_id = int(parts[-1].strip())
            item_dict[item_id] = item_title

    return item_dict


def count_item_frequencies(train_csv_file: str, item_dict: Dict[int, str]) -> Dict[str, int]:
    """
    Count the frequency of each item_id in the training data,
    and convert item_id to item_title.

    Args:
        train_csv_file (str): Path to the CSV file containing training data with item_id.
        item_dict (Dict[int, str]): Mapping from item_id to item_title.

    Returns:
        Dict[str, int]: Mapping from item_title to frequency.
    """
    item_id_counter = Counter()

    with open(train_csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            item_id = int(row['item_id'])
            item_id_counter[item_id] += 1

    item_frequency_dict = {}
    for item_id, frequency in item_id_counter.items():
        item_title = item_dict.get(item_id, "Unknown Item")
        item_frequency_dict[item_title] = frequency

    return item_frequency_dict


def save_item_frequencies_to_json(output_json_file: str, item_frequency_dict: Dict[str, int]) -> None:
    """
    Save item_title and its frequency to a JSON file.

    Args:
        output_json_file (str): Path to save the output JSON file.
        item_frequency_dict (Dict[str, int]): Mapping from item_title to frequency.
    """
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)

    with open(output_json_file, 'w', encoding='utf-8') as file:
        json.dump(item_frequency_dict, file, ensure_ascii=False, indent=4)

    print(f"[INFO] Item frequency statistics saved to: {output_json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute item frequencies from training data and save as JSON."
    )
    parser.add_argument(
        "--meta_file", type=str, required=True,
        help="Path to the item metadata file (contains item_title and item_id)"
    )
    parser.add_argument(
        "--train_file", type=str, required=True,
        help="Path to the training CSV file (contains item_id column)"
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to the output JSON file to save item_title: frequency mapping"
    )

    args = parser.parse_args()

    item_dict = load_item_titles(args.meta_file)
    item_frequency_dict = count_item_frequencies(args.train_file, item_dict)
    save_item_frequencies_to_json(args.output_file, item_frequency_dict)


if __name__ == "__main__":
    main()
