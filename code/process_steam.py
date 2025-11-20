import fire
from loguru import logger
import json
from tqdm import tqdm
import random
import time
import datetime
import csv
import os
import ast

def gao_steam(category, metadata_path=None, reviews_path=None, K=5, output=True):
    """
    Process the Steam dataset (full time range) to generate sequential recommendation data.
    This version is compatible with non-standard JSON formats and keeps the output format
    identical to the original Amazon script.

    Args:
        category (str): Prefix for output files, e.g., 'Steam_All'.
        metadata_path (str, optional): Path to Steam game metadata. Default './steam_games.json'.
        reviews_path (str, optional): Path to Steam reviews. Default './steam_reviews.json'.
        K (int, optional): K-core filter threshold. Default 5.
        output (bool, optional): Whether to write output files. Default True.
    """
    # --- 1. Data loading and preprocessing ---
    logger.info(f"Starting processing for Steam dataset (ALL DATA). Output prefix: '{category}'")

    if metadata_path is None:
        metadata_path = './steam_games.json'
    if reviews_path is None:
        reviews_path = './steam_reviews.json'

    try:
        logger.info(f"Loading and parsing reviews from {reviews_path} using ast.literal_eval...")
        with open(reviews_path, 'r', encoding='utf-8') as f:
            raw_reviews = [ast.literal_eval(line) for line in tqdm(f, desc="Parsing reviews file")]

        logger.info(f"Loading and parsing metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            try:
                raw_metadata_list = [ast.literal_eval(line) for line in tqdm(f, desc="Parsing metadata file")]
                raw_metadata = {item['id']: item for item in raw_metadata_list if 'id' in item}
            except (SyntaxError, ValueError):
                # Fallback for a single JSON-like object stored as whole content
                f.seek(0)
                content = f.read()
                raw_metadata = ast.literal_eval(content)
    except Exception as e:
        logger.error(f"An error occurred during file parsing: {e}")
        return

    logger.info("Preprocessing Steam data to a unified format...")

    # Standardized field names to align with Amazon-style schema
    user_id_col, item_id_col, time_col, rating_col, title_col = 'reviewerID', 'asin', 'unixReviewTime', 'overall', 'title'

    # Normalize reviews to the target schema
    reviews = []
    for r in tqdm(raw_reviews, desc="Processing Reviews"):
        if not all(k in r for k in ['user_id', 'product_id', 'date']):
            continue
        try:
            timestamp = int(datetime.datetime.strptime(r['date'], '%Y-%m-%d').timestamp())
            reviews.append({user_id_col: r['user_id'], item_id_col: r['product_id'], time_col: timestamp, rating_col: 1.0})
        except (ValueError, KeyError):
            continue

    # Normalize metadata to the target schema
    metadata = []
    for item_id, item_info in tqdm(raw_metadata.items(), desc="Processing Metadata"):
        if 'app_name' in item_info and item_info['app_name']:
            metadata.append({item_id_col: item_id, title_col: item_info['app_name']})

    logger.info(f"Loaded and processed metadata: {len(metadata)}, reviews: {len(reviews)}")

    # --- 2. Metadata cleaning ---
    remove_items = set()
    id_title = {}
    for meta in tqdm(metadata, desc="Cleaning metadata"):
        asin = meta[item_id_col]
        if (title_col not in meta) or not meta[title_col]:
            remove_items.add(asin)
            continue
        title = meta[title_col].replace("&quot;", "\"").replace("&amp;", "&").strip()
        if 1 < len(title) and len(title.split(" ")) <= 25:
            id_title[asin] = title
        else:
            remove_items.add(asin)

    # Ensure items appearing in reviews exist in cleaned metadata
    for review in tqdm(reviews, desc="Validating items in reviews"):
        if review[item_id_col] not in id_title:
            remove_items.add(review[item_id_col])

    # --- 3. K-core filtering ---
    remove_users = set()
    while True:
        flag = False
        users, items, new_reviews = {}, {}, []
        for review in tqdm(reviews, desc="K-core filtering iteration"):
            if review[user_id_col] in remove_users or review[item_id_col] in remove_items:
                continue
            users.setdefault(review[user_id_col], 0); users[review[user_id_col]] += 1
            items.setdefault(review[item_id_col], 0); items[review[item_id_col]] += 1
            new_reviews.append(review)
        for user, count in users.items():
            if count < K: remove_users.add(user); flag = True
        for item, count in items.items():
            if count < K: remove_items.add(item); flag = True
        density = len(new_reviews) / (len(users) * len(items)) if users and items else 0
        logger.info(f"Users: {len(users)}, Items: {len(items)}, Reviews: {len(new_reviews)}, Density: {density:.6f}")
        if not flag: break
    reviews = new_reviews
    logger.info(f"Final dataset stats: Users: {len(users)}, Items: {len(items)}, Reviews: {len(reviews)}")

    # --- 4. Build sequences, split datasets, and save ---
    items = list(items.keys())
    random.seed(42)
    random.shuffle(items)
    item2id = {}
    os.makedirs("./info", exist_ok=True)
    output_file_prefix = f"{category}_{K}-core"
    with open(f"./info/{output_file_prefix}.txt", 'w', encoding='utf-8') as f:
        for i, item in enumerate(items):
            item2id[item] = i
            f.write(f"{id_title[item]}\t{i}\n")

    if not output:
        logger.info("Output is disabled. Process finished."); return

    # Group interactions by user
    interact = {}
    for review in tqdm(reviews, desc="Grouping interactions by user"):
        user = review[user_id_col]
        interact.setdefault(user, []).append(review)

    # Generate rolling sequences per user
    interaction_list = []
    for user_id, user_reviews in tqdm(interact.items(), desc="Generating sequences"):
        sorted_reviews = sorted(user_reviews, key=lambda x: int(x[time_col]))
        item_asins = [r[item_id_col] for r in sorted_reviews]
        item_ids = [item2id[asin] for asin in item_asins]
        item_titles = [id_title[asin] for asin in item_asins]
        ratings = [r[rating_col] for r in sorted_reviews]
        timestamps = [r[time_col] for r in sorted_reviews]
        for i in range(1, len(item_asins)):
            st = max(i - 10, 0)
            # Keep the order and structure exactly the same as the original script
            interaction_list.append([
                user_id,             # user_id
                item_asins[st:i],    # item_asins
                item_asins[i],       # item_asin
                item_ids[st:i],      # history_item_id
                item_ids[i],         # item_id
                item_titles[st:i],   # history_item_title
                item_titles[i],      # item_title
                ratings[st:i],       # history_rating
                ratings[i],          # rating
                timestamps[st:i],    # history_timestamp
                timestamps[i]        # timestamp
            ])

    logger.info(f"Total sequences generated: {len(interaction_list)}")
    interaction_list.sort(key=lambda x: int(x[-1]))
    train_end_idx, valid_end_idx = int(len(interaction_list) * 0.8), int(len(interaction_list) * 0.9)
    train_data, valid_data, test_data = interaction_list[:train_end_idx], interaction_list[train_end_idx:valid_end_idx], interaction_list[valid_end_idx:]

    os.makedirs("./train", exist_ok=True); os.makedirs("./valid", exist_ok=True); os.makedirs("./test", exist_ok=True)

    # Use the exact same header as the original Amazon script
    header = ['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'history_timestamp', 'timestamp']

    for split_name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
        with open(f"./{split_name}/{output_file_prefix}.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
        logger.info(f"{split_name.capitalize()} set size: {len(data)}")

    logger.info("Processing finished successfully!")

if __name__ == '__main__':
    fire.Fire(gao_steam)
