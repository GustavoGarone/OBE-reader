# 🧠 OBE-Reader

Sistema para leitura automática de provas (OMR), usando visão computacional e modelos YOLOv8-Pose para detectar cantos, alinhar páginas e identificar respostas e IDs.
O foco é obter alta acurácia com imagens simples (scanner ou foto).

## ⚙️ Instalação

```bash
git clone <repo_url>
cd OBE-reader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Como usar

Coloque todas as provas em:
`data/00--raw/...`

`flatten_unzip.py` auxilia a padronização da pasta.

Execute os notebooks na ordem:

1. `00_clean_pipeline.ipynb`

2. `01_keypoints.ipynb`

3. `02_core.ipynb`

Após o resultado automático, `fix_manual.py`
auxilia a para corrigir manualmente possíveis erros.


## 📝 Importante:
Respeite os pedidos de inspeção manual indicados nas células dos notebooks
e ajuste os caminhos de entrada e saída conforme sua máquina.

## 📂 Estrutura esperada

```prompt
data/
├── 00--raw/ # imagens originais
├── 10--clean/ # recortes e padronização
├── 20--results/ # resultados CSV e relatórios
```


