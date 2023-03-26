from utils import swap_df_path
import os

if __name__ == '__main__':
    data_root =  '/root/autodl-tmp'
    csv_name = 'diffusiondb_15W_add_embedding.csv'
    dataset = 'dataset_15W'
    df = swap_df_path(data_root, csv_name, dataset) #csv路径变为本地

    csv_new = 'diffusiondb_15W_add_embedding_mine.csv'
    df.to_csv(os.path.join(data_root, csv_new), index=False) #保存新的df