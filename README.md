# Vida Predict
---

## 🚀 Configuração do Ambiente de Desenvolvimento


### 1. Pré-requisitos

* **Git**
* **pyenv** (e seus plugins, como `pyenv-virtualenv`)

### 2. Instruções de Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/rickfsoares/Vida-Predict.git
    cd Vida-Predict
    ```

2.  **Crie e Ative o Ambiente Virtual:**

    ```bash
    # Crie o ambiente virtual com o nome 'vida-predict'
    python -m venv vida-predict

    # Ative o ambiente (Linux/macOS)
    source vida-predict/bin/activate

    # (Se você estiver no Windows, use o seguinte comando)
    # .\vida-predict\Scripts\activate
    ```
    *Após ativar, você deve ver `(vida-predict)` no início do seu prompt.*

3.  **Instale as Dependências:**
    Todas as bibliotecas necessárias (pandas, pysus, dbfread, scikit-learn, etc.) estão no `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

---

