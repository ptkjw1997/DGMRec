import os
import json
import pandas as pd
import numpy as np
import argparse

def split_arr_(arr, k = 4) :
    res = []
    l = len(arr) / k
    check_ = 0
    for i in range(k) :
        from_, to_ = int(i * l), int((i+1) * l)
        res.append(arr[from_:to_])
        check_ += len(res[i])
    assert check_ == len(arr)
    return res

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

    np.random.seed(1225)
    # missing items are selected equally for each group : New Cold, New Warm, Old Cold, Old Warm
    missing_items_nc = np.random.choice(new_items_cold, size = int(missing_ratio * len(new_items_cold)), replace = False)
    missing_items_nw = np.random.choice(new_items_warm, size = int(missing_ratio * len(new_items_warm)), replace = False)
    missing_items_oc = np.random.choice(old_items_cold, size = int(missing_ratio * len(old_items_cold)), replace = False)
    missing_items_ow = np.random.choice(old_items_warm, size = int(missing_ratio * len(old_items_warm)), replace = False)
    missing_items = np.concatenate((missing_items_nc, missing_items_nw, missing_items_oc, missing_items_ow))


    mnc = split_arr_(missing_items_nc)
    mnw = split_arr_(missing_items_nw)
    moc = split_arr_(missing_items_oc)
    mow = split_arr_(missing_items_ow)

    missing_items_dict = {}
    missing_items_dict['t'] = np.concatenate((mnc[0], mnw[0], moc[0], mow[0]))
    missing_items_dict['v'] = np.concatenate((mnc[1], mnw[1], moc[1], mow[1]))
    missing_items_dict['all'] = np.concatenate((mnc[2], mnw[2], moc[2], mow[2],
                                                mnc[3], mnw[3], moc[3], mow[3]))

    np.save(f"{dataset_name}/missing_items_{missing_ratio_name}", missing_items_dict, allow_pickle = True)

