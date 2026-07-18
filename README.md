# DinoGesture

Projeto de reconhecimento de expressões faciais a partir dos **468 landmarks** do MediaPipe Face Mesh. O fluxo cobre a coleta de sessões pela webcam, a geração de um dataset tabular, o treinamento de um classificador e a inferência em tempo real.

## Visão geral

```text
Webcam → frames PNG por sessão → Face Mesh → 1.404 features → modelo → predição na tela
```

Cada landmark possui três coordenadas (`x`, `y`, `z`). Por isso, cada frame válido gera `468 × 3 = 1.404` features numéricas.

Antes de salvar ou enviar as features ao modelo, todos os pontos são centralizados pelo landmark `1` (ponta do nariz). Assim, o nariz fica em `(0, 0, 0)` e o modelo aprende a geometria relativa do rosto, em vez da posição da pessoa na tela.

## Estrutura do projeto

```text
DinoGesture/
├── data/
│   ├── recordings/              # Sessões próprias, organizadas por expressão
│   ├── collected_landmarks.csv  # Dataset gerado a partir das sessões
│   ├── collected_landmarks_rejected.csv
│   ├── archive/                 # Dataset original e scripts de referência
│   └── TreatedData/             # Dados já tratados para experimentos
├── model/
│   └── SVM.py                   # Carregamento e inferência do modelo salvo
├── notebooks/
│   ├── Analise.ipynb            # Explorações e tratamento inicial dos dados
│   ├── data_analisis.ipynb      # Análise estatística do dataset original
│   └── Trainings/
│       └── SVM.ipynb            # Treinamento e avaliação do classificador
├── scripts/
│   ├── capture_expression.py    # Coleta de uma sessão pela webcam
│   ├── build_landmarks_csv.py   # Geração do CSV de landmarks
│   └── script_face.py           # Visualização e extração manual com Face Mesh
├── main.py                      # Aplicação de inferência em tempo real
├── pipeline.py                  # Contexto e execução da pipeline
└── steps.py                     # Etapas de captura, detecção, predição e exibição
```

## Instalação

O projeto usa Python `3.12` e [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
```

## Criando um dataset próprio

### 1. Capture uma expressão

```powershell
python scripts/capture_expression.py
```

O coletor abre a webcam, mostra 3 segundos de preparação e então tenta gravar 120 frames em 24 FPS. Ao final, informe o nome da expressão. Uma nova sessão é salva nesta estrutura:

```text
data/recordings/
└── sorriso/
    └── 20260718T143000_a1b2c3d4/
        ├── frame_0001.png
        ├── ...
        └── frame_0120.png
```

Gravar novamente `sorriso` cria outra subpasta de sessão, sem sobrescrever as anteriores. Pressione `Q` ou `ESC` para cancelar a coleta.

### 2. Converta os frames em landmarks

```powershell
python scripts/build_landmarks_csv.py
```

O script percorre todas as sessões, exibe uma barra de progresso e recria os arquivos abaixo:

| Arquivo | Conteúdo |
| --- | --- |
| `data/collected_landmarks.csv` | Frames válidos com `label`, `session_id`, `frame_index` e 1.404 coordenadas. |
| `data/collected_landmarks_rejected.csv` | Frames sem rosto, com múltiplos rostos, ilegíveis ou com quantidade inválida de landmarks. |

O CSV principal usa este formato:

```text
label,session_id,frame_index,x_0,y_0,z_0,...,x_467,y_467,z_467
sorriso,20260718T143000_a1b2c3d4,1,...
```

`label` identifica a expressão. `session_id` identifica uma gravação completa. `frame_index` preserva a ordem do frame, mas não deve ser usado como feature do modelo.

## Treinamento

Os notebooks de treinamento ficam em `notebooks/Trainings/`. O notebook `SVM.ipynb` é o ponto de partida para carregar o CSV, treinar o classificador e salvar o modelo em formato Joblib.

Ao montar os dados para treino, use apenas as colunas de landmarks como entrada:

```python
feature_columns = [
    column for column in df.columns
    if column.startswith(("x_", "y_", "z_"))
]

X = df[feature_columns]
y = df["label"]
groups = df["session_id"]
```

### Evitando data leakage

Os 120 frames de uma sessão são muito parecidos. Portanto, nunca use `train_test_split` padrão nesses frames: ele pode colocar imagens da mesma sessão em treino e teste.

Separe os dados por `session_id` com `GroupShuffleSplit` ou `StratifiedGroupKFold`. Dessa forma, uma sessão inteira pertence a apenas um conjunto.

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_index, test_index = next(splitter.split(X, y, groups=groups))
```

Para uma avaliação confiável, grave pelo menos duas sessões por expressão; cinco ou mais é um objetivo melhor.

## Inferência em tempo real

Após treinar e salvar um modelo compatível com as 1.404 coordenadas centralizadas pelo nariz:

```powershell
python main.py
```

A pipeline executa as seguintes etapas:

```text
CaptureFrameStep
  → MirrorFrameStep
  → DetectFaceStep
  → DrawLandmarksStep
  → PredictFaceCommandStep
  → DisplayFrameStep
```

`DetectFaceStep` preserva os landmarks originais para o desenho e produz as features normalizadas para o modelo. A janela mostra a classe prevista quando houver um rosto detectado.

> O modelo usado em tempo real deve ser treinado com o mesmo formato gerado por `build_landmarks_csv.py`. Misturar dados brutos com dados centralizados pelo nariz torna as previsões inválidas.

## Tecnologias

- Python 3.12
- OpenCV
- MediaPipe Face Mesh
- pandas
- scikit-learn
- Joblib
- tqdm
