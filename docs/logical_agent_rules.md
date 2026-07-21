# Agente lógico proposicional do DinoGesture

## Classificação

O fluxo de controle do Chrome Dino é um agente baseado em conhecimento. Ele
recebe percepções da Face Mesh, representa o estado atual em sentenças
proposicionais, inclui essas sentenças em uma base de conhecimento por `TELL`,
deriva consequências por encadeamento para frente e consulta a ação por `ASK`.

Essa arquitetura separa claramente a percepção geométrica do conhecimento
declarativo e do mecanismo de inferência. A referência adotada é o capítulo 7
de [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/2nd-ed/newchap07.pdf),
de Russell e Norvig.

## Ciclo por frame

```text
Face Mesh -> medidas normalizadas -> fatos proposicionais
                                      |
                         TELL(fatos) na base de conhecimento
                                      |
                     encadeamento para frente até ponto fixo
                                      |
                          ASK(acao:*) -> ação semântica
                                      |
                         confirmação temporal -> Chrome Dino
```

1. A Face Mesh fornece landmarks da boca, olhos e sobrancelhas.
2. As distâncias são normalizadas pela distância entre os cantos externos dos
   olhos.
3. Nos primeiros 24 frames válidos, o agente usa a mediana das medidas como
   baseline do rosto neutro.
4. Depois da calibração, os limiares da KB transformam as medidas em fatos
   mutuamente exclusivos.
5. Uma KB nova recebe os fatos do frame por `TELL`; suas regras estáticas são
   aplicadas por encadeamento para frente.
6. `ASK` exige exatamente uma conclusão de ação. A mesma ação deve ocorrer em
   três frames consecutivos antes de ser enviada ao atuador.

## Vocabulário proposicional

| Grupo | Átomos |
| --- | --- |
| Boca | `boca_aberta`, `boca_fechada` |
| Sobrancelhas | `sobrancelhas_levantadas`, `sobrancelhas_nao_levantadas` |
| Ações | `acao:pular`, `acao:abaixar`, `acao:neutro` |

Os fatos negativos são representados explicitamente, em vez de assumir que a
ausência de um fato significa sua negação. Isso permite que a regra de neutro
seja uma consequência lógica da KB.

## Regras padrão

As regras ficam em `knowledge/dino_rules.json`:

```text
boca_aberta -> acao:abaixar
boca_fechada AND sobrancelhas_levantadas -> acao:pular
boca_fechada AND sobrancelhas_nao_levantadas -> acao:neutro
```

`boca_aberta` tem prioridade porque as duas últimas regras exigem
`boca_fechada`. Assim, boca aberta e sobrancelhas levantadas resultam somente
em `acao:abaixar`.

## Segurança da KB

Antes de iniciar, o carregador valida JSON, limiares positivos, formato das
regras e ações reconhecidas. Também avalia as quatro combinações possíveis de
fatos perceptivos. Cada uma deve derivar exatamente uma ação; uma KB ambígua
ou incompleta não inicia o programa.

O modo `--debug-mode` exibe os fatos de `TELL` e o resultado de `ASK`, além das
medidas usadas para produzir os fatos. Isso torna o raciocínio observável na
demonstração.
