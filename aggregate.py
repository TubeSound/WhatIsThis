import os
import pandas as pd
import numpy as np  
import glob

def main():
    files = glob.glob('./result/*.xlsx')
    for symbol in ['NIKKEI', 'DOW']:
        dfs = []
        for file in files:
            if file.find(symbol) >= 0:
                _, filename = os.path.split(file)
                values = filename.split('_')
                if len(values) < 4:
                    continue
                num = values[0]
                df = pd.read_excel(file)
                df['number'] = num
                dfs.append(df)
        df = pd.concat(dfs)
        df = df.sort_values('profit', ascending=False)
        df = df.iloc[:1000, :]
        df.to_excel('./result/' + symbol + '.xlsx', index=False)
        
    pass

if __name__ == '__main__':
    main()