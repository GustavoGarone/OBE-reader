# 🧠 OBE-Reader

Sistema para leitura automática de provas (OMR), usando visão computacional e
modelos YOLOv8-Pose para detectar cantos, alinhar páginas e identificar respostas e IDs.
O foco é obter alta acurácia com imagens simples (scanner ou foto).

## ⚙️ Instalação

```bash
git clone https://github.com/ptonso/OBE-reader.git
cd OBE-reader
pip install -r requirements.txt
```

### ❄ Para o Nix

```
nix-shell default.nix
```

Recomendamos usar o [direnv](https://wiki.nixos.org/wiki/Direnv):
```
echo "use nix" >> .envrc
```

## 🚀 Como usar

Coloque todas as provas em: `data/00--raw/...` e execute
`flatten_unzip.py data/00--raw/ data/01--cleaned/10--start/` para auxiliar na
organização das pastas.

Execute os notebooks na ordem:

1. `00_clean_pipeline.ipynb`

2. `01_keypoints.ipynb`

3. `02_core.ipynb`

Após o resultado automático, `fix_manual.py`
auxilia a para corrigir manualmente possíveis erros.


## 📝 Importante:

Respeite os pedidos de inspeção manual indicados nas células dos notebooks
e ajuste os caminhos de entrada e saída conforme sua máquina.

Recomenda-se usar GPU da NVIDIA (CUDA) para acelerar o processamento.

## 📂 Estrutura esperada

```prompt
data/
├── 00--raw/ # imagens originais
├── 10--clean/ # recortes e padronização
├── 20--results/ # resultados CSV e relatórios
```


