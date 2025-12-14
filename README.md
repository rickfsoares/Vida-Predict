# 🩺 Vida Predict: Predição de Risco de Diabetes Tipo 1 via SUS

> Um sistema de alerta precoce baseado em Machine Learning e dados públicos do DATASUS.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-green)
![Status](https://img.shields.io/badge/Status-Data%20Preparation%20Complete-orange)

---

## 📖 Sobre o Projeto

O **Vida Predict** é um projeto de Ciência de Dados aplicado à Saúde Pública. O objetivo é desenvolver um modelo de classificação capaz de identificar **perfis de alto risco** para Diabetes Mellitus Tipo 1 (DM1) utilizando dados administrativos do SUS.

Diferente de um diagnóstico clínico tradicional, nossa abordagem utiliza dados históricos de pacientes já diagnosticados para treinar um algoritmo capaz de **calcular a probabilidade de um novo paciente ser um portador não diagnosticado da doença** ou possuir características biológicas estatisticamente idênticas ao grupo de risco.

### 🎯 Objetivo Principal
Criar uma ferramenta de **triagem ativa** que analise grandes volumes de dados (Peso, Altura, Idade, Histórico Familiar) para alertar gestores de saúde sobre pacientes que necessitam de monitoramento prioritário.

---

## 📊 Fonte de Dados

Os dados foram extraídos do **DATASUS** (Departamento de Informática do SUS), focando no Estado de São Paulo (2022). Utilizamos duas fontes principais anonimizadas:

1.  **APAC de Medicamentos (AMSP):** Fonte primária para extração de Rótulos (CID E10 - Diabetes Tipo 1) e features antropométricas (Peso e Altura).
2.  **Boletim de Produção Ambulatorial (BISP):** Fonte secundária para engenharia de features, especificamente para identificar **Histórico Familiar de Diabetes** (CID Z833).

---

## 📂 Estrutura do Projeto

```text
Vida-Predict/
│
├── etl/                  # Módulo de Engenharia de Dados
│   ├── extractor.py      # Leitura eficiente dos arquivos Parquet
│   └── transformer.py    # Limpeza, Engenharia de Features e Regras de Negócio
│
├── dados-unificados/     # Datasets processados (Final: dataset_final_modelo.parquet)
├── config.py             # Configurações globais e constantes
├── pipeline.py           # Script orquestrador principal
├── analise_exploratoria.py # Script para validação estatística e busca de outliers
└── README.md             # Documentação do projeto
```

---

## 🚀 Como Executar o Projeto

1. Pré-requisitos

- Git
- Python 3.8+
- Virtualenv (recomendado)

2. Instalar as dependências

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/rickfsoares/Vida-Predict.git
cd Vida-Predict

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate # Windows

# Instalar dependências
pip install -r requirements.txt
```

3. Executando o pipeline
```bash
python pipeline.py
```

---

## 🔗 Links Úteis

* 📓 **Notebook de Análise Exploratória:** [Acessar Google Colab](https://colab.research.google.com/drive/1CvLV5s-HWXrIGAKqGAnFVo0uACIY6aTy?usp=sharing)

* 📄 **Artigo Científico de Referência:** [Visualizar PDF](https://sol.sbc.org.br/index.php/sbcas/article/view/16082)

---

## 👥 Autores

* **[Ricardo França Soares](https://github.com/rickfsoares)**
* **[Gabriel Oliveira](https://github.com/BielzinDaAgua)**
* **[Caio Bernardelli](https://github.com/CaioBernardelli)**
* **[Flavio Nascimento](https://github.com/FlavioNascimento99)**

Desenvolvido como parte da disciplina de Tópicos Especiais.
