## Offensive Language Identification and Meme Classification in dravidian languages
This repository contains all the work related to the Offensive Language Identification and Meme Classification  tasks organized by dravidianlangtech2021.
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

In order to reproduce the results obtained you can clone this repository and place ur dataset path in the  train scripts to run the same.

Our results for the Offensive Language Identification Task

![image](https://user-images.githubusercontent.com/42090593/118062526-b9f10e00-b3b4-11eb-8885-9d23344dd615.png)

Our results for the Meme Classification Task

![image](https://user-images.githubusercontent.com/42090593/118062623-edcc3380-b3b4-11eb-83d2-b814f5b19927.png)

If you find our work useful do consider citing our paper.
```
@inproceedings{ghanghor-etal-2021-iiitk,
    title = "{IIITK}@{D}ravidian{L}ang{T}ech-{EACL}2021: Offensive Language Identification and Meme Classification in {T}amil, {M}alayalam and {K}annada",
    author = "Ghanghor, Nikhil  and
      Krishnamurthy, Parameswari  and
      Thavareesan, Sajeetha  and
      Priyadharshini, Ruba  and
      Chakravarthi, Bharathi Raja",
    booktitle = "Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages",
    month = apr,
    year = "2021",
    address = "Kyiv",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.dravidianlangtech-1.30",
    pages = "222--229",
    abstract = "This paper describes the IIITK team{'}s submissions to the offensive language identification, and troll memes classification shared tasks for Dravidian languages at DravidianLangTech 2021 workshop@EACL 2021. Our best configuration for Tamil troll meme classification achieved 0.55 weighted average F1 score, and for offensive language identification, our system achieved weighted F1 scores of 0.75 for Tamil, 0.95 for Malayalam, and 0.71 for Kannada. Our rank on Tamil troll meme classification is 2, and offensive language identification in Tamil, Malayalam and Kannada are 3, 3 and 4 respectively.",
}
```
