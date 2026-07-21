# DinoGesture

Controle do Chrome Dino por gestos faciais. O modo principal é um **agente
lógico proposicional**: ele percebe o rosto, registra fatos em uma base de
conhecimento, infere uma ação e a executa no jogo.

```text
Webcam -> Face Mesh -> medidas normalizadas -> fatos
                                             |
                                    TELL -> KB -> inferência -> ASK
                                             |
                                      pular | abaixar | neutro
```

O projeto também mantém, em `test_model_webcam.py`, um experimento separado
com Random Forest. Esse experimento é aprendizado de máquina; ele não integra
o agente lógico que controla o jogo.

## Instalação

O projeto usa Python 3.12 e `uv`.

```powershell
uv sync
```

## Executar

Visualize a percepção e a ação inferida:

```powershell
python main.py
```

Controle o Chrome Dino com o rosto:

```powershell
python main.py --game-control
```

Para a demonstração, o modo abaixo mostra medidas, fatos enviados por `TELL` e
a ação respondida por `ASK`:

```powershell
python main.py --game-control --debug-mode
```

## Base de conhecimento

As regras e os limiares vivem em `knowledge/dino_rules.json`. Ela é carregada
e validada antes de abrir a câmera ou o Chrome. Uma KB alternativa pode ser
usada sem editar o código:

```powershell
python main.py --game-control --rules caminho\para\regras.json
```

O formato é JSON nativo:

```json
{
  "mouth_threshold": 0.08,
  "brow_threshold": 0.04,
  "rules": [
    {"if": ["boca_aberta"], "then": "acao:abaixar"},
    {"if": ["boca_fechada", "sobrancelhas_levantadas"], "then": "acao:pular"},
    {"if": ["boca_fechada", "sobrancelhas_nao_levantadas"], "then": "acao:neutro"}
  ]
}
```

A KB deve inferir exatamente uma ação para cada combinação válida dos fatos de
percepção. Arquivos ausentes, JSON inválido, ações desconhecidas, regras
ambíguas ou incompletas interrompem a inicialização com uma mensagem clara.

Leia a descrição completa em [docs/logical_agent_rules.md](docs/logical_agent_rules.md).

## Dataset e modelo experimental

Os scripts em `scripts/` capturam expressões e convertem 468 landmarks em
1.404 coordenadas para treinamento. Os notebooks fazem a análise e o
treinamento, separando sessões por `session_id` para evitar *data leakage*.

```powershell
python scripts/capture_expression.py
python scripts/build_landmarks_csv.py
python test_model_webcam.py
```

## Referência

O desenho do agente segue o modelo de agentes baseados em conhecimento de
Russell e Norvig: uma base de conhecimento recebe sentenças (`TELL`) e produz
respostas inferidas a consultas (`ASK`). [Artificial Intelligence: A Modern
Approach, capítulo 7](https://aima.cs.berkeley.edu/2nd-ed/newchap07.pdf).
