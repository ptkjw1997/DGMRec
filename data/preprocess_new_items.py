import os
import json
import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby')
    args = parser.parse_args()

    dataset_name = args.dataset
    missing_ratio = 2/3
    missing_ratio_name = 0.666 # For convinience of file name 

    df = pd.read_csv(f"{dataset_name}/{dataset_name}.inter", sep = '\t')
    n_items = df['itemID'].nunique()

    df1 = df[df['x_label'] == 0]
    item_cnt_df = pd.DataFrame(data = {'itemID' : np.arange(n_items)})
    item_cnt_df = item_cnt_df.join(df1[['userID', 'itemID']].groupby('itemID').count())
    item_cnt_df.rename(columns = {'userID' : 'train_cnt'}, inplace = True)
    item_cnt_df.fillna(0, inplace = True)
    item_cnt_df.sort_values('train_cnt', inplace = True)

    items_group_cold = np.array(item_cnt_df[:int(n_items * 0.7)]['itemID'].tolist())
    items_group_warm = np.array(item_cnt_df[int(n_items * 0.7):]['itemID'].tolist())

    new_item_ratio = 0.2

    np.random.seed(1111)
    new_items_cold = np.random.choice(items_group_cold, size = int(new_item_ratio * len(items_group_cold)), replace = False)
    new_items_warm = np.random.choice(items_group_warm, size = int(new_item_ratio * len(items_group_warm)), replace = False)
    new_items = np.concatenate((new_items_cold, new_items_warm))

    all_items = np.arange(n_items)
    old_items = np.setdiff1d(all_items, new_items)
    old_items_cold = np.setdiff1d(items_group_cold, new_items_cold)
    old_items_warm = np.setdiff1d(items_group_warm, new_items_warm)

    np.save(f"{dataset_name}/new_items.npy", new_items)
    np.save(f"{dataset_name}/cold_items.npy", items_group_cold)
    np.save(f"{dataset_name}/warm_items.npy", items_group_warm)



    uid_field = 'userID'
    iid_field = 'itemID'
    split = 'x_label'

    cols = [uid_field, iid_field, split]
    ratio = 0.2

    df = pd.read_csv(f"{dataset_name}/{dataset_name}.inter", usecols=cols, sep="\t")

    item_num = int(max(df[iid_field].values)) + 1
    user_num = int(max(df[uid_field].values)) + 1

    dfs = []
    for i in range(3):
        temp_df = df[df[split] == i].copy()
        dfs.append(temp_df)
    train_u = set(dfs[0][uid_field].values)
    
    # Loading New Items Index / Removing New items in train/valid
    new_items = np.load(f"{dataset_name}/new_items.npy")
    dfs[2] = pd.concat([dfs[2], dfs[0][dfs[0]['itemID'].isin(new_items)]])
    dfs[0] = dfs[0][~dfs[0]['itemID'].isin(new_items)]

    train_df, valid_df, test_df = dfs
    
    
    df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)

    df.to_csv(f"{dataset_name}/{dataset_name}_del.inter", sep='\t', index=False)
        