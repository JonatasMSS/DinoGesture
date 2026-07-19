# Agente lógico de comandos faciais

## Objetivo

O agente lógico transforma medidas geométricas da Face Mesh em três comandos simples, sem depender de um classificador treinado:

| Comando | Significado | Regra principal |
| --- | --- | --- |
| `0` | Neutro | Nenhuma regra de ativação foi atingida. |
| `1` | Boca aberta | A abertura da boca aumentou em relação ao rosto neutro. |
| `2` | Sobrancelhas levantadas | As duas sobrancelhas subiram em relação ao rosto neutro. |

O fluxo é: **webcam → Face Mesh → medidas normalizadas → calibração → regras → confirmação temporal → comando**.

## Pontos faciais utilizados

O MediaPipe Face Mesh fornece 468 landmarks. O agente usa apenas os pontos abaixo:

| Medida | Landmarks | Finalidade |
| --- | --- | --- |
| Escala do rosto | `33` e `263` | Cantos externos dos olhos; normaliza as distâncias. |
| Abertura da boca | `13` e `14` | Lábio superior e lábio inferior. |
| Sobrancelha esquerda | `70`, `63`, `105`, `66`, `107` | Calcula a posição média da sobrancelha esquerda. |
| Referência do olho esquerdo | `159`, `145` | Calcula o centro vertical do olho esquerdo. |
| Sobrancelha direita | `336`, `296`, `334`, `293`, `300` | Calcula a posição média da sobrancelha direita. |
| Referência do olho direito | `386`, `374` | Calcula o centro vertical do olho direito. |

## Normalização geométrica

Para reduzir o impacto de a pessoa estar mais perto ou mais longe da câmera, todas as medidas são divididas pela distância entre os cantos externos dos olhos:

```text
escala_olhos = distância_2D(ponto_33, ponto_263)
```

Assim, o agente trabalha com proporções do rosto, e não com pixels ou posições absolutas na tela.

### Abertura da boca

```text
abertura_boca = distância_2D(ponto_13, ponto_14) / escala_olhos
```

Quanto maior esse valor, maior a distância entre os lábios.

### Elevação das sobrancelhas

Primeiro é calculada a média vertical (`y`) dos pontos de cada sobrancelha e o centro vertical de cada olho. Em seguida:

```text
elevacao_esquerda = (centro_y_olho_esquerdo - media_y_sobrancelha_esquerda) / escala_olhos
elevacao_direita  = (centro_y_olho_direito  - media_y_sobrancelha_direita)  / escala_olhos
```

Na imagem, o eixo `y` cresce para baixo. Portanto, quando a sobrancelha sobe, seu `y` diminui e a distância vertical para o olho aumenta. Um valor maior representa sobrancelha mais levantada.

## Calibração do rosto neutro

Antes de reconhecer comandos, o sistema coleta **24 frames válidos** com a pessoa em expressão neutra.

Para cada medida, o baseline é a mediana das 24 amostras:

```text
baseline_boca             = mediana(abertura_boca)
baseline_sobrancelha_esq  = mediana(elevacao_esquerda)
baseline_sobrancelha_dir  = mediana(elevacao_direita)
```

A mediana reduz o efeito de pequenos ruídos da câmera e de movimentos isolados. Se nenhum rosto for encontrado, a calibração apenas pausa; os frames já coletados não são perdidos.

Depois da calibração, o agente usa a diferença entre a medida atual e o baseline:

```text
delta = medida_atual - baseline
```

Isso torna a decisão relativa ao rosto da própria pessoa, em vez de usar uma distância fixa igual para todos.

## Regras de decisão

Os limiares iniciais são configuráveis e conservadores:

| Condição | Limiar padrão | Comando |
| --- | --- | --- |
| `delta_boca >= 0.08` | `--mouth-threshold 0.08` | `1` — boca aberta |
| `delta_sobrancelha_esq >= 0.04` **e** `delta_sobrancelha_dir >= 0.04` | `--brow-threshold 0.04` | `2` — sobrancelhas levantadas |
| Nenhuma das condições | — | `0` — neutro |

A boca aberta tem prioridade. Portanto, se a boca estiver aberta e as duas sobrancelhas também estiverem levantadas no mesmo frame, o resultado é o comando `1`.

```text
se delta_boca >= limiar_boca:
    decisao = 1
senão se delta_sobrancelha_esq >= limiar_sobrancelha
        e delta_sobrancelha_dir >= limiar_sobrancelha:
    decisao = 2
senão:
    decisao = 0
```

## Estabilidade temporal

Uma decisão não muda o comando imediatamente. Ela precisa aparecer em **3 frames consecutivos** para ser confirmada.

Em uma câmera a 24 FPS, isso equivale aproximadamente a 125 ms. Esse filtro evita que ruídos, piscadas de detecção ou microvariações da Face Mesh gerem comandos instáveis.

```text
se decisao_atual == decisao_candidata:
    incrementar_contador
senão:
    decisao_candidata = decisao_atual
    contador = 1

se contador >= 3:
    comando = decisao_candidata
```

## Casos especiais

- Sem rosto detectado: não há comando (`None`) e a interface mostra “Sem face detectada”.
- Durante a calibração: a interface mostra “Calibrando... mantenha o rosto neutro”.
- Distância entre os olhos igual a zero: a medida é descartada para evitar divisão por zero.
- O comando anterior só é mantido enquanto uma nova decisão ainda não tiver sido confirmada pelos 3 frames.

## Modo de depuração visual

Para exibir os pontos e as medidas usadas pelo agente:

```powershell
python main.py --show-logic-points
```

Na janela são desenhados:

- pontos da boca em vermelho;
- pontos de referência dos olhos em azul;
- pontos das sobrancelhas em amarelo;
- linhas da abertura da boca e da escala entre os olhos;
- deltas normalizados de boca e sobrancelhas.

Os limiares podem ser ajustados na execução:

```powershell
python main.py --mouth-threshold 0.08 --brow-threshold 0.04
```

## Vantagens e limitações

**Vantagens:** funciona sem dataset ou treinamento, é explicável, rápido e permite ajustar os limiares em tempo real.

**Limitações:** reconhece somente os três comandos definidos; depende de uma calibração neutra adequada; grandes rotações do rosto, oclusões ou iluminação ruim podem afetar os landmarks. Os limiares devem ser refinados com a câmera e a distância de uso reais.

## Resumo para o slide

> O sistema mede a abertura da boca e a distância entre sobrancelhas e olhos, normaliza as medidas pela distância entre os olhos, calibra o rosto neutro e só confirma um comando após três frames consecutivos. Assim, produz comandos interpretáveis: neutro (`0`), boca aberta (`1`) e duas sobrancelhas levantadas (`2`).
