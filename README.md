## Offensive Language Identification and Meme Classification in dravidian languages
This repository contains all the work related to the Offensive Language Identification and Meme Classification  tasks organized by dravidianlangtech.
`The requirements to run this codes are:`
1. pytorch
2. transformers
3. sadice
4. seaborn
5. sklearn
6. matplotlib

We have used the pretrained transformers BERT , IndicBERT and XLM-Roberta for the ``Offensive Language Identification task``. Along with the pretrained transformers original versions we have used customized versions of these models as well.
The customized versions were being built by freezing the base layers and adding a fc layer on the top of it with custom loss functions :
`nll_loss` and `sadice loss`.

For the ``Meme Classification task`` , we have dealt with the two modalities `image` and `texts` separately.
The vision based models were built by using the pretrained inceptionv3 and resnet50 models.
The text based models were built by using the pretrained transformers BERT , IndicBERT and XLM-Roberta.
We have used the pretrained transformers BERT , IndicBERT and XLM-Roberta.
Along with the pretrained transformers original versions we have used customized versions of these models as well.
