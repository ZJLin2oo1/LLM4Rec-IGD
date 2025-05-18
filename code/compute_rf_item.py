import fire
import numpy as np
from tqdm import tqdm
import json
import os


def extract_titles(category, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct metadata file path
    metadata_file = f"meta_{category}.json"

    # Create output file path
    output_file = os.path.join(output_folder, f"{category}_statistics.json")

    id_title = {}
    with open(metadata_file, 'r') as f:
        # Reduce tqdm update frequency by setting miniters
        for line in tqdm(f, desc=f"Processing {category}", miniters=100):
            meta = json.loads(line)
            if 'title' not in meta or meta['title'].find('<span id') > -1:
                continue
            title = meta["title"].replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")
            if len(title) > 1 and len(title.split(" ")) <= 20:
                if title not in id_title:
                    id_title[title] = 1

    with open(output_file, 'w') as f:
        json.dump(id_title, f, indent=4)

    print(f"Total unique titles extracted: {len(id_title)}")
    print(f"Results saved to {output_file}")


def main():
    fire.Fire(extract_titles)

if __name__ == "__main__":
    main()