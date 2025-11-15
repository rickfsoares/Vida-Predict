from pysus import SIA

sia = SIA().load()

# APAC de Medicamentos
files = sia.get_files("AM", uf="SP", year=2022)

# Boletim de Produção Ambulatorial individualizado
# files = sia.get_files("BI", uf="SP", year=2022)

# APAC de Laudos Diversos
# files = sia.get_files("AD", uf="SP", year=2022)

sia.download(files, local_dir="./dados-datasus-processados/")
