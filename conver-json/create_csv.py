def csv(df, output_name):
    csv_filename = output_name
    df.to_csv('../landmark-csv/'+csv_filename+'.csv', index=False)