# Waste Classifier

Classificador de imagens de lixo em três classes (orgânico, reciclável e não reciclável), feito com **transfer learning** a partir da MobileNetV2. O foco do projeto é entender o que muda ao reaproveitar um modelo já treinado e como lidar com **classes desbalanceadas**, e não só chegar a um número de acurácia.

## Dataset

[Waste Classification Data v2](https://www.kaggle.com/datasets/sapal6/waste-classification-data-v2) (Kaggle), com duas pastas, `TRAIN` e `TEST`, cada uma com três classes:

| Classe | Significado | TRAIN | TEST |
|--------|-------------|-------|------|
| `N` | não reciclável | 2.847 (12,6%) | 397 (13,6%) |
| `O` | orgânico | 12.565 (55,7%) | 1.401 (48,1%) |
| `R` | reciclável | 7.152 (31,7%) | 1.112 (38,2%) |

O significado de `N`, `O` e `R` foi inferido das descrições de versões anteriores do dataset (confirmar na página do Kaggle).

Observações sobre os dados:

- A classe `N` tem só ~13% das imagens. Acurácia sozinha esconderia erros nela, por isso a avaliação usa precision, recall e matriz de confusão por classe.
- No `TRAIN`, todos os arquivos são `.jpg`. No `TEST` da classe `N`, 3 dos 397 arquivos (`.webp`, `.cfm` e `.img`) não são lidos pelo Keras e são ignorados sem aviso.
- O dataset não tem validação. A validação sai de 15% de cada classe do `TRAIN`, e o `TEST` original fica intocado até o resultado final.

## Abordagem

1. **Amostra primeiro.** Antes de treinar no dataset completo (22 mil imagens, em CPU), o fluxo foi testado em uma amostra balanceada (400 imagens por classe no treino, 100 na validação e 100 no teste). Isso permitiu medir o tempo por epoch (uns 13 s na amostra) e achar problemas baratos, como o formato `.webp`.
2. **Transfer learning.** MobileNetV2 com pesos do ImageNet, sem a camada final original, com a base congelada e uma camada nova (`GlobalAveragePooling2D`, `Dropout(0,3)` e `Dense(3, softmax)`). Dos 2,26 milhões de parâmetros, só 3.843 aprendem nessa fase.
3. **Fine-tuning.** Liberação das últimas 30 camadas da base (mantendo as camadas de normalização congeladas), com taxa de aprendizado de `1e-5`.
4. **Desbalanceamento.** Peso por classe no treino (`N` ≈ 2,64, `O` ≈ 0,60 e `R` ≈ 1,05).
5. **Augmentation só no treino** (rotação de até 20° e espelhamento horizontal). Validação e teste usam só o pré-processamento da MobileNetV2.

## Resultados

Validação do dataset completo (3.386 imagens, proporção real das classes):

| Modelo | Acurácia | `N` precision | `N` recall | `R` recall |
|--------|----------|---------------|------------|------------|
| Base congelada, com pesos de classe | 85% | 0,53 | 0,90 | 0,78 |
| Mesmo modelo, probabilidades divididas pelos pesos¹ | 89% | 0,67 | 0,82 | 0,84 |
| Após fine-tuning | [...] | [...] | [...] | [...] |

¹ É uma aproximação de "como seria sem pesos", e não substitui um treino sem pesos.

**Teste final** (`TEST` original, avaliado uma vez): [...]

## O que aprendi

- **A amostra balanceada esconde o desbalanceamento.** Na amostra, a `N` tinha precision de 78% a 84%. No dataset completo, com os pesos de classe, caiu para 53%.
- **O peso de classe pode ficar forte demais.** Com `N` pesando ~2,6 vezes, a rede passou a chamar de `N` muita coisa que não era: dos 728 previstos como `N`, 341 estavam errados. Tirar o efeito do peso reduziu os falsos `N` pela metade e melhorou as outras classes.
- **A principal confusão é entre `N` e `R`.** Apareceu na amostra balanceada e continuou no completo. Das imagens de `R` que a rede chamou de `N`, 7 de 8 que olhei eram mesmo recicláveis, então o limite parece estar mais no modelo do que nos rótulos.
- **O pré-processamento precisa ser o do modelo.** A MobileNetV2 espera pixels entre -1 e 1, e não entre 0 e 255. Com a escala errada, nada dá erro, mas o resultado piora.
- **Um bug do ambiente:** o gerador do teste estava lendo a pasta de validação, e só foi notado porque a contagem de imagens (3.386) coincidia com a da validação.

## Limitações

- Treino em CPU, com poucas epochs. Mais epochs e um ajuste de hiperparâmetros poderiam melhorar o resultado.
- O dataset foi montado a partir de buscas na web, então as imagens não representam fotos tiradas em um ponto de coleta real.
- A classe `N` parece misturar itens parecidos com `R` (como plásticos), o que explica parte da confusão.
- Um erro tem custo diferente do outro: reciclável jogado no lixo comum desperdiça material, e não reciclável colocado na reciclagem contamina o lote. O critério de qual pesa mais depende do uso, e não foi definido aqui.

## Estrutura

```
waste-classifier/
├── waste_classifier.ipynb   # notebook principal
├── data/                    # dataset (não versionado)
├── modelos/                 # modelos treinados (não versionado)
└── README.md
```

## Como executar

1. Instale as dependências: `pip install tensorflow matplotlib scikit-learn`
2. Baixe o dataset do Kaggle e coloque em `data/waste-classification-data-v2`.
3. Rode o notebook em ordem. As células que copiam arquivos (amostra e divisão treino/validação) só precisam rodar uma vez.

No Windows, o TensorFlow atual não usa GPU, então o treino roda em CPU (cerca de 4 a 5 minutos por epoch no dataset completo).