# Grape Frame Extractor

Ferramenta para extração estruturada de frames representativos a partir de vídeos de vinhedo, desenvolvida como parte do Trabalho de Conclusão de Curso em Engenharia de Computação no IFMG *Campus* Bambuí.

## Contexto

Os vídeos foram capturados na plantação experimental do Instituto Politécnico de Bragança (IPB), em Portugal, e fazem parte do *dataset* utilizado para o desenvolvimento de um modelo de detecção e estimativa de produção de uvas baseado em aprendizado profundo.

O objetivo deste script é ampliar o conjunto de imagens disponíveis para treinamento extraindo frames dos vídeos de forma criteriosa, evitando redundância e priorizando qualidade visual.

## Como funciona

O algoritmo divide cada vídeo em janelas temporais correspondentes a 4 janelas por segundo. Para cada janela, seleciona o frame mais adequado com base em dois critérios:

1. **Nitidez** — medida pela variância do operador Laplaciano aplicado ao frame em escala de cinza. Frames mais nítidos produzem maior variância.
2. **Luminosidade** — medida pela intensidade média dos pixels. Frames com luminosidade abaixo do limiar configurável são descartados como candidatos principais.

**Critério de seleção:**
- Se houver frames com luminosidade adequada na janela, seleciona o mais nítido entre eles.
- Se todos os frames da janela estiverem abaixo do limiar de luminosidade, seleciona o mais nítido do lote independentemente da iluminação (*fallback*).

Esse critério garante exatamente um frame por janela, independentemente das condições do vídeo.

## Requisitos

- Python 3.8 ou superior
- OpenCV
- NumPy

Instalação das dependências:

```bash
pip install opencv-python numpy
```

## Uso

Coloque os arquivos de vídeo na pasta raiz do projeto com os nomes `video1.mp4` até `video7.mp4` e execute:

```bash
python extract_frames.py
```

Os frames extraídos serão salvos em `dataset_uvas_estruturado/<nome_do_video>/`.

### Configurações

As configurações principais estão no início do arquivo `extract_frames.py`:

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `FRAMES_POR_SEGUNDO` | `4` | Número de janelas por segundo |
| `LIMIAR_LUMINOSIDADE` | `40.0` | Luminosidade mínima (0–255) para considerar um frame bem iluminado |
| `PASTA_SAIDA_BASE` | `dataset_uvas_estruturado` | Pasta raiz de saída |

## Estrutura de saída

```
dataset_uvas_estruturado/
├── video1/
│   ├── frame_000000_orig_000000.jpg
│   ├── frame_000001_orig_000015.jpg
│   └── ...
├── video2/
│   └── ...
└── ...
```

O nome de cada arquivo indica o índice do frame no conjunto extraído (`frame_XXXXXX`) e o índice do frame original no vídeo (`orig_XXXXXX`).

## Projeto relacionado

Este script faz parte de um *pipeline* maior que inclui:

- Filtragem de redundância por *Perceptual Hashing* (pHash)
- Anotação do *dataset* na plataforma Roboflow
- Treinamento de modelo de segmentação de instâncias baseado na família YOLO
- Algoritmo de estimativa quantitativa de produção de uvas

## Autor

Hugo César Leal  
Engenharia de Computação — IFMG *Campus* Bambuí  
Orientador: Prof. Dr. Ciniro Aparecido Leite Nametala  
Coorientador: Prof. Dr. João Carlos Sampaio Maldonado Costa Mendes (UPB, Portugal)
