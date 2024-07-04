import os
import pandas as pd
import numpy as np  
import glob

def main():
    files = glob.glob('./2018.10-2024.5/result/*.xlsx')
    for symbol in ['NIKKEI', 'DOW', 'NSDQ']:
        dfs = []
        for file in files:
            if file.find(symbol) >= 0:
                _, filename = os.path.split(file)
                values = filename.split('_')
                if len(values) < 4:
                    continue
                num = values[0]
                try:
                    df = pd.read_excel(file, engine="openpyxl")
                except:
                    print('error', file)
                    continue
                df['number'] = num
                if len(df) > 0:
                    dfs.append(df)
        if len(dfs) == 0:
            continue
        df = pd.concat(dfs)
        df = df.sort_values('profit', ascending=False)
        df = df.iloc[:1000, :]
        df.to_excel('./2018.10-2024.5/result/' + symbol + '.xlsx', index=False)
        
    pass

if __name__ == '__main__':
    main()