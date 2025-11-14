from pysus import SIA
import pandas as pd

sia = SIA().load()

files = sia.get_files("PA", uf="SP", year=2022)

sia.download(files, local_dir="./dados-datasus-processados/")

#parquet = sia.download(files, local_dir="./dados-datasus-processados/")[0]

#print(parquet.to_dataframe())


#df = pd.read_parquet("./dados-datasus-processados/PASP0001.parquet/")

#df = pd.read_parquet("./dados-datasus-processados/PASP1801a.parquet/")
#print(df.head())
