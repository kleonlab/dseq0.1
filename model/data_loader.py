def prepare_split(df, min_cells=0, max_cells_train=2000):
    # 1. Remove the "Tail" (Too noisy)
    df_clean = df[df['Cell Count'] >= min_cells].copy()
    print(f"Dropped {len(df) - len(df_clean)} genes due to low cell counts.")
    
    # 2. Stratified Split
    # We use the Gene Name to ensure every gene is in both train and test
    train_df, test_df = train_test_split(
        df_clean, 
        test_size=0.15, 
        stratify=df_clean['Gene Name'],
        random_state=42
    )
    
    # 3. Cap the "Head" in Training only (Optional but recommended)
    # This prevents PINK1 from dominating the training gradients
    train_df_capped = train_df.groupby('Gene Name').apply(
        lambda x: x.sample(n=min(len(x), max_cells_train), random_state=42)
    ).reset_index(drop=True)
    
    return train_df_capped, test_df

# Usage
train_set, test_set = prepare_split(df)

1. check the type of file handling - like what is df, it needs to go from h5ad 
2. ensure that the output is a dictionary with two different parts: training, testing each different dictionary 
3. the sequence needs to be tokenized before getting saved in the dictionary 
4. fix the control cell state handling - perhaps get some inspiration from the cell-load package 
5. use a config file like the cell-load package to give information into the function 
6. add this function in the training loop file. 