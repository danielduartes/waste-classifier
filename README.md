# Waste Classifier

Classificador de imagens de lixo em três classes (`N`, `O` e `R`), feito com **transfer learning** a partir da MobileNetV2 (Keras/TensorFlow). O projeto acabou sendo menos sobre o número de acurácia e mais sobre uma pergunta: **o que o modelo está realmente aprendendo?** A investigação do resultado mostrou que os rótulos do dataset não concordam com o critério comum de reciclável, e isso muda a forma de ler todos os números.

## Resumo dos resultados

- **91%** de acurácia na validação e **78%** no teste original.
- Esses números medem o quanto o modelo **reproduz o rótulo do dataset**, e não o quanto ele acerta reciclabilidade.
- A classe `O` (orgânico) é a única em que os números têm significado direto: na amostra que rotulei à mão, o rótulo do dataset concordou comigo em 20 de 20 imagens.
- Nas pastas `N` e `R` do teste, o rótulo do dataset discordou do meu critério em 40 de 40 imagens da amostra, com respostas espelhadas (o que o dataset chama de `N`, eu chamei de `R`, e vice-versa).
- Em 19 imagens sorteadas de `TRAIN/N`, só 1 era `N` pelo meu critério.

## Dataset

[Waste Classification Data v2](https://www.kaggle.com/datasets/sapal6/waste-classification-data-v2) (Kaggle), com duas pastas, `TRAIN` e `TEST`, cada uma com três classes:

| Classe | Significado | TRAIN | TEST |
|--------|-------------|-------|------|
| `N` | não reciclável | 2.847 (12,6%) | 397 (13,6%) |
| `O` | orgânico | 12.565 (55,7%) | 1.401 (48,1%) |
| `R` | reciclável | 7.152 (31,7%) | 1.112 (38,2%) |

O significado de `N`, `O` e `R` foi inferido das descrições de versões anteriores do dataset (a confirmar na página do Kaggle).

Observações sobre os dados:

- A classe `N` tem só ~13% das imagens. Acurácia sozinha esconderia erros nela, por isso a avaliação usa precision, recall e matriz de confusão por classe.
- No `TRAIN`, todos os arquivos são `.jpg`. No `TEST` da classe `N`, 3 dos 397 arquivos (`.webp`, `.cfm` e `.img`) não são lidos pelo Keras e são ignorados sem aviso, por isso o teste avalia 2.907 das 2.910 imagens.
- O dataset não tem validação. A validação sai de 15% de cada classe do `TRAIN`, e o `TEST` original fica intocado até o resultado final.

## Abordagem

1. **Amostra primeiro.** Antes de treinar no dataset completo (22 mil imagens, em CPU), o fluxo foi testado em uma amostra balanceada (400 imagens por classe no treino, 100 na validação e 100 no teste). Isso permitiu medir o tempo por epoch (uns 13 s na amostra) e achar problemas baratos, como o formato `.webp`.
2. **Transfer learning.** MobileNetV2 com pesos do ImageNet, sem a camada final original, com a base congelada e uma camada nova (`GlobalAveragePooling2D`, `Dropout(0,3)` e `Dense(3, softmax)`). Dos 2,26 milhões de parâmetros, só 3.843 aprendem nessa fase. Imagens redimensionadas para 160×160, lotes de 32.
3. **Fine-tuning.** Liberação das últimas 30 camadas da base (mantendo as camadas de normalização congeladas), com taxa de aprendizado de `1e-5`. O melhor modelo foi salvo pelo `ModelCheckpoint` com base na `val_accuracy` (epoch 3 de 5).
4. **Desbalanceamento.** Peso por classe no treino (`N` ≈ 2,64, `O` ≈ 0,60 e `R` ≈ 1,05).
5. **Augmentation só no treino** (rotação de até 20° e espelhamento horizontal). Validação e teste usam só o pré-processamento da MobileNetV2 (pixels entre -1 e 1).
6. **Disciplina de avaliação.** O teste foi usado uma vez, com o modelo e a versão (normal, sem ajuste de probabilidades) escolhidos antes de ver o resultado dele.

## Resultados

### Amostra (teste do fluxo)

Na amostra balanceada, a base congelada chegou a ~81% de acurácia na validação (300 imagens), e o fine-tuning a ~82%. Com uma validação desse tamanho, a diferença está dentro do ruído, e o ganho do fine-tuning só ficou claro no dataset completo.

### Validação do dataset completo (3.386 imagens, proporção real das classes)

| Modelo | Acurácia | `N` precision | `N` recall | `R` recall |
|--------|----------|---------------|------------|------------|
| Base congelada, com pesos de classe | 85% | 0,53 | 0,90 | 0,78 |
| Mesmo modelo, probabilidades divididas pelos pesos¹ | 89% | 0,67 | 0,82 | 0,84 |
| Fine-tuning (melhor epoch) | 91% | 0,72 | 0,87 | 0,89 |
| Fine-tuning, probabilidades divididas pelos pesos¹ | 92% | 0,83 | 0,78 | 0,91 |

¹ Aproximação de "como seria sem pesos de classe", e não um treino sem pesos. Cada versão troca um tipo de erro pelo outro: a normal erra menos ao chamar `N` de `R` (43 casos) e mais ao chamar `R` de `N` (101); a ajustada faz o contrário (68 e 53).

A versão **normal** foi escolhida para o teste, antes de ver o resultado dele. A validação vem de dentro do `TRAIN`, e a epoch do fine-tuning foi escolhida por ela, então esses números são levemente otimistas.

### Teste (`TEST` original, 2.907 imagens lidas, avaliado uma vez)

| Classe | Precision | Recall |
|--------|-----------|--------|
| `N` | 0,48 | 0,81 |
| `O` | 0,88 | 0,97 |
| `R` | 0,86 | 0,54 |

Acurácia de **78%**. Matriz de confusão (linhas: classe real; colunas: prevista):

| Real \ Previsto | N | O | R |
|---|---|---|---|
| **N** | 319 | 7 | 68 |
| **O** | 13 | 1.357 | 31 |
| **R** | 335 | 174 | 603 |

A queda de 13 pontos em relação à validação vem quase toda da classe `R`: 335 das 1.112 imagens foram classificadas como `N`.

## Investigação do resultado do teste

Em vez de aceitar os 78% como "o modelo é pior do que parecia", investiguei de onde vinha a diferença.

**1. Grades de imagens.** Entre 24 imagens de `R` do `TEST` que o modelo classificou como `N`, quase nenhuma é reciclável pelo critério comum: cortinas, lenços e papel higiênico, canudos plásticos, papel de seda. Entre 24 imagens de `N` do `TRAIN`, a maioria é papelão, latas, garrafas PET e sacolas. Ou seja, o modelo errava em imagens cujo rótulo parecia errado.

**2. Rotulagem manual, às cegas.** Sorteei 80 imagens (20 de cada pasta do `TEST` e 20 do `TRAIN/N`), embaralhadas, sem ver o rótulo do dataset nem a previsão do modelo. 79 ficaram válidas: 60 do `TEST` e 19 do `TRAIN/N`.

Critério informal, formado durante a rotulagem e não registrado por escrito antes: `R` é o que costuma ir na coleta seletiva (papelão, latas, garrafas); `O` é matéria orgânica; `N` é o que não costuma ser aceito (lenço e papel higiênico usados, cortinas, canudos). Sacolas plásticas foram o caso duvidoso, e respondi pelo que vi na imagem.

| Pasta do dataset | Imagens | O que eu respondi |
|---|---|---|
| `TEST/N` | 20 | `R` em 20 |
| `TEST/O` | 20 | `O` em 20 |
| `TEST/R` | 20 | `N` em 20 |
| `TRAIN/N` | 19 | concordei com o rótulo em 1 |

O rótulo do dataset concordou com meu critério em 33% das imagens do `TEST` (20 de 60, todas da pasta `O`), e o modelo concordou em 47%. Nas pastas `N` e `R` do teste, as minhas respostas foram exatamente o espelho dos rótulos. Para checar que não havia troca de colunas na gravação, conferi à mão amostras da pasta `R` (papel higiênico, canudos e cortinas, todos respondidos como `N`).

**3. Cinco imagens de fora do dataset** (fundo simples), para ver o que o modelo responde em dados de outra fonte:

| Item | Esperado (meu critério) | Resposta do modelo |
|---|---|---|
| Caixa de papelão | `R` | `R` 0,66 / `N` 0,33 |
| Banana | `O` | `O` 0,98 |
| Copo descartável sujo | `N` | `O` 0,50 / `R` 0,49 |
| Canudo | `N` | **`R` 0,96** |
| Cortina | `N` | `O` 0,42 / `R` 0,41 / `N` 0,17 |

O modelo acertou 2 de 5. O canudo é o caso mais claro: ele aprendeu que canudo é `R` com 96% de confiança, e é o que o rótulo do dataset diz. A banana (`O`, 98%) confirma que a classe orgânica está bem definida. Cortina e copo ficaram quase empatados entre classes, porque o modelo não tem a opção "isto não é lixo".

## O que aprendi

- **Rótulo e realidade são coisas diferentes.** Um modelo com 91% na validação pode estar reproduzindo um rótulo que não corresponde ao que o nome da classe sugere. Só olhando as imagens, e não as métricas, isso apareceu.
- **Validação e teste de fontes diferentes dão números diferentes.** A validação saiu do próprio `TRAIN`, então compartilha as convenções de rótulo dele. O teste veio de outra pasta, e a diferença de 13 pontos foi o sinal.
- **A amostra balanceada esconde o desbalanceamento.** Na amostra, a `N` tinha precision de 78% a 84%. No dataset completo, com pesos de classe, caiu para 53%.
- **O peso de classe pode ficar forte demais.** Com `N` pesando ~2,6 vezes, a rede chamou de `N` muita coisa que não era (341 falsos `N` entre 728 previstos). Tirar o efeito do peso reduziu isso pela metade.
- **O fine-tuning ajudou de verdade.** A validação foi de 85% para ~90%, e `R` previsto como `N` caiu de 210 para 101.
- **O pré-processamento precisa ser o do modelo.** A MobileNetV2 espera pixels entre -1 e 1, e não entre 0 e 255. Com a escala errada, nada dá erro, mas o resultado piora.
- **Conferir a contagem de imagens pega bugs silenciosos.** O gerador do teste leu a pasta de validação, e só foi notado porque a contagem (3.386) era igual à da validação. Formatos de arquivo que o Keras não lê também somem sem aviso.
- **Decidir antes de olhar.** Escolher o modelo e a versão antes de ver o teste é o que permite chamar aquele número de honesto.

## Limitações

- **O dataset v2 tem rótulos inconsistentes nas classes `N` e `R`**, pelo critério usado aqui. Não sei como o autor os montou. A soma `7.152 (R) + 2.847 (N) = 9.999` é igual ao tamanho da classe reciclável do dataset original de duas classes, o que sugere que a `N` foi separada da `R`, mas é só uma inferência.
- **A rotulagem manual é de uma pessoa só**, com amostra pequena (60 do teste e 19 do treino): a margem de erro é de uns 10 a 15 pontos. Outra pessoa rotularia diferente nos itens ambíguos (sacola, papel usado). Os resultados mais firmes são o do `TRAIN/N` e o espelhamento nas pastas `N` e `R`.
- **A rotulagem não é independente da hipótese.** Já conhecia o padrão pelas grades de imagens antes de rotular. Ela foi às cegas quanto ao rótulo de cada imagem, mas não quanto à suspeita geral.
- **O significado de `N`, `O` e `R`** foi inferido de descrições de versões anteriores do dataset.
- **O modelo só conhece três classes**, e não tem a opção "isto não é lixo".
- **Treino em CPU**, com 5 epochs por fase. Mais epochs poderiam melhorar o resultado (a validação ainda oscilava entre epochs).
- **As versões "ajustadas pelos pesos"** são uma aproximação, e não um treino sem pesos.
- **No `TEST` da classe `N`, 3 de 397 arquivos** (`.webp`, `.cfm` e `.img`) não são lidos pelo Keras.
- **As imagens do dataset e as imagens de teste não estão no repositório.** Os achados são descritos em texto.

## Próximos passos possíveis

- Refazer o treino sem pesos de classe (ou com pesos mais suaves, como a raiz do peso) para comparar de verdade com a aproximação.
- Repetir o projeto com um dataset de rótulos por material (por exemplo, o Garbage Classification, com papelão, vidro, metal, papel, plástico e lixo comum), em que o rótulo é menos ambíguo.
- Rotulagem manual por mais de uma pessoa, para medir a concordância entre critérios.
- Adicionar uma classe "não é lixo" para casos como cortina e copo.

## Estrutura

```
waste-classifier/
├── waste_classifier.ipynb   # notebook principal
├── data/                    # dataset (não versionado)
├── modelos/                 # modelos treinados e rótulos manuais (não versionado)
├── imagens_teste/           # imagens de teste próprias (não versionado)
└── README.md
```

O notebook segue esta ordem: preparação e contagem dos dados, tratamento das imagens, amostra, dataset completo (treino da camada final e fine-tuning), teste final, investigação dos erros, rotulagem manual e imagens de fora do dataset.

## Como executar

1. Instale as dependências: `pip install tensorflow matplotlib scikit-learn ipython`
2. Baixe o dataset do Kaggle e coloque em `data/waste-classification-data-v2`.
3. Rode o notebook em ordem, seção por seção. As células que copiam arquivos (amostra e divisão treino/validação) só precisam rodar uma vez, e o fine-tuning salva o melhor modelo em `modelos/modelo_c_fase2.keras`.
4. Os sorteios usam semente fixa (`random.seed`), então as amostras se repetem.

No Windows, o TensorFlow atual não usa GPU, então o treino roda em CPU (cerca de 4 a 5 minutos por epoch no dataset completo, e uns 18 minutos para o fine-tuning de 5 epochs).