# Malu

Malu AI Source.

## Estrutura

- malu/: codigo do modelo, dataset, treino e geracao
- scripts/: scripts utilitarios
- data/processed/: dados pre-processados
- checkpoints/: checkpoints do treino

## Setup

pip install -r requirements.txt

## Preparar dados

python scripts/prepare_data.py

## Treinar

bash scripts/train.sh

## Gerar

python -m malu.generate --prompt "Ola, quem e voce?"
