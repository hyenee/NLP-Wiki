# ๐๏ธ Research ๊ด๋ จ github ์ ๋ฆฌ

DL, NLP ์ฐ๊ตฌ ๊ด๋ จ ์ ์ฉํ github์ ์ ๋ฆฌํฉ๋๋ค.

*โ๏ธ Last Update : 2022.02.05*

## ๐ข Index
- [๐๏ธ Research ๊ด๋ จ github ์ ๋ฆฌ](#๏ธ-research-๊ด๋ จ-github-์ ๋ฆฌ)
  - [๐ข Index](#-index)
  - [Crawler](#crawler)
  - [Dataset](#dataset)
  - [Dataset(Ko)](#datasetko)
  - [Generation](#generation)
  - [Infra](#infra)
  - [Korean NLP](#korean-nlp)
  - [Latex / Notion / README](#latex--notion--readme)
  - [Libraries / Toolkits](#libraries--toolkits)
  - [Metric](#metric)
  - [Model](#model)
  - [Survey](#survey)
  - [Streamlit](#streamlit)
  - [Subtask](#subtask)
  - [Tutorials](#tutorials)
  - [Wiki](#wiki)



## Crawler
`ํฌ๋กค๋ฌ`
| Name                                                                          | Description                                                                     |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [Web_Crawler](https://github.com/BitnaKeum/Web_Crawler)                       | ๋๋ฌด์ํค, ์ํคํผ๋์, ๋ค์๋ธ๋ก๊ทธ, ํฐ์คํ ๋ฆฌ, ์ ํ๋ธ, ๋ค์ดํธํ ํฌ๋กค๋ฌ                 |
| [youtube_crawler](https://github.com/Mo0nl19ht/youtube_crawler)               | ์ ํ๋ธ API๋ฅผ ์ด์ฉํ ํค์๋์ ๋ฐ๋ฅธ ์ ํ๋ธ ์์ URL, ์ ๋ชฉ, ์์ธ์ ๋ณด, ๋๊ธ, ์๋ง ํฌ๋กค๋ง |
| [Youtube_Comment_Crawler](https://github.com/SOMJANG/Youtube_Comment_Crawler) | ์ ํ๋ธ ๋๊ธ ํฌ๋กค๋ฌ                                                                |
| [naver-blog-crawler](https://github.com/xotrs/naver-blog-crawler)             | ๋ค์ด๋ฒ ๋ธ๋ก๊ทธ ํฌ์คํ URL, ํฌ์คํ ์ ๋ชฉ, ํฌ์คํ ์ค๋ช, ํฌ์คํ ๋ ์ง, ๋ธ๋ก๊ฑฐ ์ด๋ฆ, ํฌ์คํ ๋ด์ฉ ํฌ๋กค๋ง |


## Dataset
`NLP ๊ด๋ จ ๋ฐ์ดํฐ์`
| Name                                                                                  | Description                                                       |
|---------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [slot_filling_and_intent_detection_of_SLU](https://github.com/sz128/slot_filling_and_intent_detection_of_SLU) | intent/slot ๋ฐ์ดํฐ์ ์ ๋ฆฌ (ATIS, SNIPS, Facebookโs multilingual dataset, MIT_Restaurant_Movie_corpus, E-commerce Shopping Assistant (ECSA) from Alibaba, CoNLL-2003 NER) |
| [rasa-nlu-benchmark](https://github.com/nghuyong/rasa-nlu-benchmark)                  | Rasa NLU ๋ฐ์ดํฐ์ (ATIS, SNIPS, AskUbuntuCorpus, Facebook Multilingual Task Oriented Dataset, SMP2019, Check flow dataset, MSRA_NER, ToutiaoNews)  |
| [simulated-dialogue](https://github.com/google-research-datasets/simulated-dialogue) | Machines Talking To Machines (M2M) ๋ฐ์ดํฐ์ (Sim-M, Sim-R, Sim-Gen) |
| [NLU-Evaluation-Data](https://github.com/xliuhw/NLU-Evaluation-Data)                 | NLU ๋ฐ์ดํฐ์ (human-robot interaction in home domain)               |
| [nlp-datasets](https://github.com/niderhoff/nlp-datasets)                            | NLP ๋ฐ์ดํฐ์                                                                                                                                                     |
| [nlp-collections](https://github.com/hyunwoongko/nlp-collections)                    | NLP ํ์คํฌ ๋ฐ์ดํฐ์ ๋ชฉ๋ก                                             |
| [paws](https://github.com/google-research-datasets/paws)                             | PAWS: Paraphrase Adversaries from Word Scrambling ๋ฐ์ดํฐ์          |
| [e2e-dataset](https://github.com/tuetschek/e2e-dataset)                              | E2E Challenge ๋ฐ์ดํฐ์                                              |


## Dataset(Ko)
`ํ๊ตญ์ด ๋ฐ์ดํฐ์`
| Name                                                                            | Description                           |
|---------------------------------------------------------------------------------|---------------------------------------|
| [AwesomeKorean_Data](https://github.com/songys/AwesomeKorean_Data)              | ํ๊ตญ์ด ๋ฐ์ดํฐ์ ๋ชฉ๋ก                   |
| [Open-korean-corpora](https://github.com/ko-nlp/Open-korean-corpora)            | ํ๊ตญ์ด ๋ฐ์ดํฐ์ ๋ชฉ๋ก                   |
| [Korpora](https://github.com/ko-nlp/Korpora)                                    | ํ๊ตญ์ด ๋ฐ์ดํฐ์ ์คํ์์ค ํ์ด์ฌ ํจํค์ง  |
| [UD_Korean-GSD](https://github.com/UniversalDependencies/UD_Korean-GSD)         | ํ๊ตญ์ด ์์กด๊ตฌ๋ฌธ๋ถ์ ๋ฐ์ดํฐ์            |
| [ud-korean](https://github.com/emorynlp/ud-korean) | ํ๊ตญ์ด ์์กด๊ตฌ๋ฌธ๋ถ์ ๋ฐ์ดํฐ์ |
| [KorNLU Datasets](https://github.com/kakaobrain/KorNLUDatasets)                | KorNLI, KorSTS ๋ฐ์ดํฐ์                 |
| [paraKQC](https://github.com/warnikchow/paraKQC)                               | ํ๊ตญ์ด parallel ๋ฐ์ดํฐ์ (10,000 utterances, namely 1,000 sets of 10 similar sentences)   |
| [korean-parallel-corpora](https://github.com/jungyeul/korean-parallel-corpora) | ํ๊ตญ์ด parallel ๋ฐ์ดํฐ์                |
| [korean-hate-speech](https://github.com/kocohub/korean-hate-speech)            | ํ๊ตญ์ด ํ์ค ๋ฐ์ดํฐ์                    |
| [nsmc](https://github.com/e9t/nsmc)                                            | ๋ค์ด๋ฒ ์ํ๋ฆฌ๋ทฐ ๊ฐ์ฑ๋ถ์ ๋ฐ์ดํฐ์ (Naver sentiment movie corpus)  |
| [KorAdvMRSTestData](https://github.com/kakaoenterprise/KorAdvMRSTestData)      | ํ๊ตญ์ด ์คํ ๋๋ฉ์ธ ๋ํ ๋ต๋ณ ์ ํ ๋ชจ๋ธ์ ์ทจ์ฝ์ ์ ํ๊ฐํ๊ธฐ ์ํ ๋ฐ์ดํฐ์ EMNLP 2021 โAn Evaluation Dataset and Strategy for Building Robust Multi-turn Response Selection Modelโ ๋ผ๋ฌธ |


## Generation
`Text Generation/Data Augmentation ๊ด๋ จ ์ฐ๊ตฌ ์ ๋ฆฌ`
| Name                                                                                    | Description                                                    |
|-----------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [nlpaug](https://github.com/makcedward/nlpaug)                                          | NLP ๊ด๋ จ augmentation ํจํค์ง                                    |
| [contextual_augmentation](https://github.com/pfnet-research/contextual_augmentation)    | NAACL 2018 โContextual Augmentation: Data Augmentation by Words with Paradigmatic Relationsโ ๋ผ๋ฌธ ๊ตฌํ |
| [Seq2SeqDataAugmentationForLU](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) | COLING 2018 โSequence-to-sequence Data Augmentation for Dialogue Language Understandingโ ๋ผ๋ฌธ ๊ตฌํ          |
| [Texygen](https://github.com/geek-ai/Texygen)                                           | SIGIR 2018 โTexygen: A Benchmarking Platform for Text Generation Modelsโ ๋ผ๋ฌธ ๊ตฌํ                         |
| [eda_nlp](https://github.com/jasonwei20/eda_nlp)                                        | MNLP-IJCNLP 2019 โEDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasksโ ๋ผ๋ฌธ ๊ตฌํ |
| [cbert_aug](https://github.com/1024er/cbert_aug)                                        | ICCS 2019 "Conditional BERT Contextual Augmentationโ ๋ผ๋ฌธ ๊ตฌํ  |
| [TransformersDataAugmentation](https://github.com/varunkumar-dev/TransformersDataAugmentation) | AACL 2020 โData Augmentation Using Pre-trained Transformer Modelsโ ๋ผ๋ฌธ ๊ตฌํ (Basline : EDA, Backtranslation, CBERT)   |
| [GPT-GNN](https://github.com/acbull/GPT-GNN)                                            | KDD 2020 โGPT-GNN: Generative Pre-Training of Graph Neural Networksโ ๋ผ๋ฌธ ๊ตฌํ                              |
| [PPLM](https://github.com/uber-research/PPLM)                                           | ICLR 2020 โPlug and Play Language Models: A Simple Approach to Controlled Text Generationโ ๋ผ๋ฌธ ๊ตฌํ        |
| [SentAugment](https://github.com/facebookresearch/SentAugment)                          | NAACL 2021 โSelf-training Improves Pre-training for Natural Language Understandingโ ๋ผ๋ฌธ ๊ตฌํ          |
| [text-autoaugment](https://github.com/lancopku/text-autoaugment)                        | EMNLP 2021 โText AutoAugment: Learning Compositional Augmentation Policy for Text Classificationโ ๋ผ๋ฌธ ๊ตฌํ    |
| [Paraphrase-Generator](https://github.com/Vamsi995/Paraphrase-Generator)                | Paraphrase Generator ๋ชจ๋ธ (dataset : Googleโs PAWS, model : T5, demo : streamlit/flask)                 |
| [T5-paraphrase-generation](https://github.com/renatoviolin/T5-paraphrase-generation)    | Paraphrase Generator ๋ชจ๋ธ (model : T5, demo : flask)  |
| [ctrl-sum](https://github.com/salesforce/ctrl-sum)                                      | ICLR 2021 โCTRLsum: Towards Generic Controllable Text Summarizationโ ๋ผ๋ฌธ ๊ตฌํ                             |
| [question_generation](https://github.com/patil-suraj/question_generation)               | Neural question generation using transformers    |
| [textaugment](https://github.com/dsfsi/textaugment)                                     | Text Augmentation ๋ผ์ด๋ธ๋ฌ๋ฆฌ (paper : โImproving Short Text Classification Through Global Augmentation Methodsโ) |



## Infra
`๋ฅ๋ฌ๋ ํ๊ฒฝ ๊ตฌ์ถ`
| Name                                                             | Description                                                      |
|------------------------------------------------------------------|------------------------------------------------------------------|
| [gpustat](https://github.com/wookayin/gpustat)                   | GPU ๋ชจ๋ํฐ๋ง ํด                                                   |
| [deepo](https://github.com/ufoym/deepo)                          | ๋ฅ๋ฌ๋ ํ๊ฒฝ ๊ตฌ์ถ์ ์ํ ๋์ปค ์ด๋ฏธ์ง ์ ๊ณต                           |
| [_Book_k8sInfra](https://github.com/sysnet4admin/_Book_k8sInfra) | โ์ปจํ์ด๋ ์ธํ๋ผ ํ๊ฒฝ ๊ตฌ์ถ์ ์ํ ์ฟ ๋ฒ๋คํฐ์ค/๋์ปคโ ์ฑ ์ค์ต ์ฝ๋ ์ ๊ณต |



## Korean NLP
`ํ๊ตญ์ด NLP ํ๋ก์ ํธ`
| Name                                                                                                | Description                                                 |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [KoEDA](https://github.com/toriving/KoEDA)                                                          | ํ๊ตญ์ด EDA (Easy Data Augmentation) ๋ชจ๋ธ                     |
| [KorEDA](https://github.com/catSirup/KorEDA)                                                        | ํ๊ตญ์ด EDA ๋ชจ๋ธ                                              |
| [Sentence_BERT_Korean](https://github.com/robinsongh381/Sentence_BERT_Korean)                       | Sentence-BERT ํ๊ตญ์ด ๋ชจ๋ธ ๊ตฌํ (dataset : KorNLI, model : DistilKoBERT) |
| [KoSentenceBERT-ETRI](https://github.com/BM-K/KoSentenceBERT-ETRI)                                  | Sentence-BERT ํ๊ตญ์ด ๋ชจ๋ธ ๊ตฌํ (dataset : KorNLU, model : ETRI KorBERT)      |
| [KoSentenceBERT-SKT](https://github.com/BM-K/KoSentenceBERT-SKT)                                    | Sentence-BERT ํ๊ตญ์ด ๋ชจ๋ธ ๊ตฌํ (dataset : KorNLU, model : SKT KoBERT)       |
| [NLP_Koeran_DP](https://github.com/hanjanghoon/NLP_Koeran_DP)                                       | ํ๊ตญ์ด ์์กด๊ตฌ๋ฌธ๋ถ์ ๋ชจ๋ธ ๊ฐ๋ฐ (2019 ๊ตญ์ด๊ฒฝ์ง๋ํ ํ๊ตญ์ด ์์กด๊ตฌ๋ฌธ ๋ถ์ ๋์)     |
| [PyKoSpacing](https://github.com/haven-jeon/PyKoSpacing)                                            | ํ๊ตญ์ด ๋์ด์ฐ๊ธฐ ๋ชจ๋ธ                                          |
| [KoBART-chatbot](https://github.com/haven-jeon/KoBART-chatbot)                                      | ํ๊ตญ์ด KoBART ์ฑ๋ด ๋ชจ๋ธ                                       |
| [KoBART-summarization](https://github.com/seujung/KoBART-summarization)                             | ํ๊ตญ์ด KoBART ์์ฝ ๋ชจ๋ธ                                       |
| [t5-summarization](https://github.com/seujung/t5-summarization)                                     | ํ๊ตญ์ด T5 ์์ฝ ๋ชจ๋ธ                                           |
| [KoGPT2-DINO](https://github.com/soeque1/KoGPT2-DINO)                                               | ํ๊ตญ์ด DINO(Generating Datasets with Pretrained Language Models ๋ผ๋ฌธ) ๋ชจ๋ธ    |
| [KoSimCSE-SKT](https://github.com/BM-K/KoSimCSE-SKT)                                                | ํ๊ตญ์ด SimCSE ๋ชจ๋ธ (paper : SimCSE: Simple Contrastive Learning of Sentence Embeddings) (dataset : KorNLI, model : SKT KoBERT) |
| [WellnessConversation-LanguageModel](https://github.com/nawnoes/WellnessConversation-LanguageModel) | AI ํ๋ธ ์ ์ ๊ฑด๊ฐ ์๋ด ๋ฐ์ดํฐ๋ฅผ ํ์ฉํ ์ฌ๋ฆฌ์๋ด ๋ํ ๋ชจ๋ธ (KoBERT, KoELECTRA, KoGPT2)  |
| [klue-transformers-tutorial](https://github.com/Huffon/klue-transformers-tutorial)                  | KLUE ๋ฐ์ดํฐ๋ฅผ ํ์ฉํ HuggingFace Transformers ํํ ๋ฆฌ์ผ (NLI, STS, zero shot classification) |
| [Awesome-Korean-NLP-Papers](https://github.com/changukshin/Awesome-Korean-NLP-Papers)              | ํ๊ตญ์ด NLP ๋ผ๋ฌธ ๋ชฉ๋ก (~2019๋๋๊น์ง ํ๊ตญํํ ์์ฃผ๋ก ์ ๋ฆฌ)        |
| [hama-py](https://github.com/hamanlp/hama-py)                                                      | ํ์ด์ฌ ํ๊ธ ์ฒ๋ฆฌ ๋ผ์ด๋ธ๋ฌ๋ฆฌ (ํํ์ ๋ถ์, ํ์ฌ ํ๊น)                                                                           |
| [hangul-toolkit](https://github.com/bluedisk/hangul-toolkit)                                       | ํ๊ธ ์๋ชจ ๋ถ๋ฆฌ/์กฐํฉ ์์์ ์ํ ํดํท                            |
| [kss](https://github.com/hyunwoongko/kss)                                                          | ํ๊ตญ์ด ๋ฌธ์ฅ ๋ถ๋ฆฌ ๋ผ์ด๋ธ๋ฌ๋ฆฌ (Kss: A Toolkit for Korean sentence segmentation)                                                  |
| [korean-sentence-splitter](https://github.com/likejazz/korean-sentence-splitter)                   | ํ๊ตญ์ด ๋ฌธ์ฅ ๋ถ๋ฆฌ ๋ผ์ด๋ธ๋ฌ๋ฆฌ                                     |
| [ko_lm_dataformat](https://github.com/monologg/ko_lm_dataformat)                                   | ํ๊ตญ์ด ์ธ์ด๋ชจ๋ธ์ฉ ํ์ต ๋ฐ์ดํฐ๋ฅผ ์ ์ฅ, ๋ก๋ฉํ๊ธฐ ์ํ ์ ํธ๋ฆฌํฐ (zstandard, ultrajson ์ ์ฌ์ฉํ์ฌ ๋ฐ์ดํฐ ๋ก๋ฉ ๋ฐ ์์ถ ์๋ ๊ฐ์ )  |
| [kospeech](https://github.com/sooftware/kospeech)                                                  | ํ๊ตญ์ด ์์ฑ ์ธ์ ํดํท (paper : Open-source toolkit for end-to-end Korean speech recognition)  |
| [Awesome-Korean-NLP](https://github.com/datanada/Awesome-Korean-NLP)                               | ํ๊ตญ์ด NLP ๊ด๋ จ ์ ๋ฆฌ                                           |
| [awesome-hangul](https://github.com/lqez/awesome-hangul)                                           | ํ๊ธ/ํ๊ตญ์ด ์ฒ๋ฆฌ ๊ด๋ จ ๋ผ์ด๋ธ๋ฌ๋ฆฌ ๋ฐ ๋ชจ๋ ๋ชฉ๋ก                    |


## Latex / Notion / README
`Latex, Notion, README์ ์ฌ์ฉํ๊ธฐ ์ข์ ํํ๋ฆฟ ์ ๋ฆฌ`
| Name                                                                                                    | Description                                             |
|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| [Notion-to-GitHub-Pages](https://github.com/uoneway/notion-to-github-pages)                             | ๋ธ์ ํ์ด์ง๋ฅผ Github Pages ๋ธ๋ก๊ทธ์ ๋ง๊ฒ ์๋ก๋ํ๋ shell script |
| [notion-py](https://github.com/jamalex/notion-py)                                                       | ๋น๊ณต์ notion API                                       |
| [Best-README-Template](https://github.com/othneildrew/Best-README-Template)                             | README ํํ๋ฆฟ ์ ๋ฆฌ                                      |
| [README-template.md ](https://github.com/scottydocs/README-template.md)                                 | README ํํ๋ฆฟ (Prerequisites, Installing, Using ๊ด๋ จ)   |
| [markdown-badges](https://github.com/Ileriayo/markdown-badges)                                          | markdown ๋ฑ์ง ๋ชฉ๋ก                                      |
| [github-readme-stats](https://github.com/anuraghazra/github-readme-stats/blob/master/docs/readme_kr.md) | readme stats ํ์                                       |
| [productive-box](https://github.com/maxam2017/productive-box)                                           | commit ์๊ฐ ํต๊ณ ๋ธ์ถ                                    |
| [awesome-pinned-gists](https://github.com/matchai/awesome-pinned-gists)                                 | A collection of awesome dynamic pinned gists for GitHub |
| [Awesome-CV](https://github.com/posquit0/Awesome-CV)                                                    | CV Latex ํํ๋ฆฟ                                         |
| [machine-learning-cheat-sheet](https://github.com/soulmachine/machine-learning-cheat-sheet)             | machine learning cheat sheet(ML ๊ฐ๋ ์ ๋ฆฌ) Latex ํํ๋ฆฟ  |


## Libraries / Toolkits
`ํ์ฉํ๊ธฐ ์ข์ ํจํค์ง/ํดํท`
| Name                                                                                | Description                                                             |
|--------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [lightning-transformers](https://github.com/PyTorchLightning/lightning-transformers) | Transformers SOTA ๋ชจ๋ธ ๊ฐ์ ธ๋ค ์ธ ์ ์๋ ํจํค์ง                          |
| [happy-transformer](https://github.com/EricFillion/happy-transformer)                | SOTA NLP ๋ชจ๋ธ ํ์ฉํ  ์ ์๋ ํจํค์ง (Text Generation, Text Classification, Word Prediction, Question Answering, Text-to-Text, Next Sentence Prediction, Token Classification) |
| [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)         | Transformers ๋ชจ๋ธ train/evaluate ํ  ์ ์๋ ํจํค์ง                       |
| [haystack](https://github.com/deepset-ai/haystack)                                   | Transformer ๋ชจ๋ธ ํ์ฉํ  ์ ์๋ ํจํค์ง (Neural Search, Question Answering, Semantic Document Search, Summarization)   |
| [flair](https://github.com/flairNLP/flair)                                           | SOTA NLP ๋ชจ๋ธ ํ์ฉํ  ์ ์๋ ํจํค์ง                                      |
| [autonlp](https://github.com/huggingface/autonlp)                                    | SOTA NLP ๋ชจ๋ธ ํ์ฉํ  ์ ์๋ ํจํค์ง                                      |
| [notebooks](https://github.com/huggingface/notebooks)                                | Huggingface ๋ผ์ด๋ธ๋ฌ๋ฆฌ notebooks                                        |
| [NeMo](https://github.com/NVIDIA/NeMo)                                               | conversational AI ํดํท (Automatic Speech Recognition, Natural Language Processing, and Text-to-Speech Synthesis)  |
| [texar-pytorch](https://github.com/asyml/texar-pytorch)                              | Machine Learning, Natural Language Processing, Text Generation ์ง์ํ๊ธฐ ์ํ ํดํท, ML ๋ชจ๋ ๋ฐ ๋ผ์ด๋ธ๋ฌ๋ฆฌ ์ ๊ณต             |
| [claf](https://github.com/naver/claf)                                               | CLaF: Open-Source Clova Language Framework                               |
| [graph4nlp](https://github.com/graph4ai/graph4nlp)                                  | NLP์์์ Graph Neural Networks ์ฝ๊ฒ ์ฌ์ฉํ๊ธฐ ์ํ ํจํค์ง                                      |
| [transformers-interpret](https://github.com/cdpierse/transformers-interpret)        | Huggingface Transformers ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ํ์ฉํ์ฌ ์ฝ๊ฒ ๋ชจ๋ธ์ ์ค๋ชํ  ์ ์๋ ํจํค์ง (Sequence Classification, Zero Shot Classification, Question Answering)      |
| [self-attentive-parser](https://github.com/nikitakit/self-attentive-parser)         | 11๊ฐ ์ธ์ด์ ๋ํ constituency parsing ํจํค์ง (SpaCy benepar Berkeley Neural Parser)                                     |
| [yanmtt](https://github.com/prajdabre/yanmtt)                                       | Yet Another Neural Machine Translation ํดํท                                        |
| [OpenPrompt](https://github.com/thunlp/OpenPrompt)                                  | An Open-Source Framework for Prompt-Learning                           |
| [oslo](https://github.com/tunib-ai/oslo)                                            | Open Source framework for Large-scale transformer Optimization                                |
| [bertviz](https://github.com/jessevig/bertviz)                                      | ํธ๋์คํฌ๋จธ ๋ชจ๋ธ ์๊ฐํ ACL 2019 โA Multiscale Visualization of Attention in the Transformer Modelโ ๋ผ๋ฌธ ๊ตฌํ                 |
| [summvis](https://github.com/robustness-gym/summvis)                                | Summarization ๋ถ์ ํด ACl 2021 "SummVis: Interactive Visual Analysis of Models, Data, and Evaluation for Text Summarizationโ ๋ผ๋ฌธ ๊ตฌํ |
| [lassl](https://github.com/lassl/lassl)                                             | Easy Language Model Pretraining leveraging Huggingface's Transformers and Datasets                                    |
| [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) | PyTorch metric learning ๊ตฌํ (TripletMarginLoss, ...)                   |
| [pytorch-summary](https://github.com/sksq96/pytorch-summary)                        | Keras์ model.summary()์ ์ ์ฌํ ํจ์ PyTorch๋ก ๊ตฌํ                     |
| [iterative-stratification](https://github.com/trent-b/iterative-stratification)     | scikit-learn cross validators (MultilabelStratifiedKFold, RepeatedMultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit) |



## Metric
`NLP metric`
| Name                                                                     | Description                                                                     |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [NLPMetrics](https://github.com/gcunhase/NLPMetrics)                     | NLP ํ๊ฐ ๋งคํธ๋ฆญ(BLEU, GLEU, WER, METEOR, TER, ROUGE, CIDEr)                      |
| [nlg-eval](https://github.com/Maluuba/nlg-eval)                          | Natural Language Generation ์๋ ํ๊ฐ ๋งคํธ๋ฆญ (BLEU, METEOR, ROUGE, CIDEr, SkipThought cosine similarity, Embedding Average cosine similarity, Vector Extrema cosine similarity, Greedy Matching score) |
| [glge](https://github.com/microsoft/glge)                                | ACL 2021 โGLGE: A New General Language Generation Evaluation Benchmarkโ ๋ผ๋ฌธ ๊ตฌํ |
| [dialoglue](https://github.com/alexa/dialoglue)                          | DialoGLUE: A Natural Language Understanding Benchmark for Task-Oriented Dialogue  |
| [bert_score](https://github.com/Tiiiger/bert_score)                      | ICLR 2020 โBERTScore: Evaluating Text Generation with BERTโ ๋ผ๋ฌธ ๊ตฌํ             |
| [BARTScore](https://github.com/neulab/BARTScore)                         | NIPS 2021 โBARTScore: Evaluating Generated Text as Text Generationโ ๋ผ๋ฌธ ๊ตฌํ     |
| [sacrebleu](https://github.com/mjpost/sacrebleu)                         | Proceedings of the Third Conference on Machine Translation 2018 โA Call for Clarity in Reporting BLEU Scoresโ ๋ผ๋ฌธ ๊ตฌํ |
| [mlm-scoring](https://github.com/awslabs/mlm-scoring)                    | ACL 2020 โMasked Language Model Scoringโ ๋ผ๋ฌธ ๊ตฌํ                                |
| [sentence-similarity](https://github.com/tuzhucheng/sentence-similarity) | Paraphrase Detection, Semantic Texual Similarity, Natural Language Inference / Textual Entailment, Answer Selection ๊ด๋ จ ๋งคํธ๋ฆญ |
| [testSignificanceNLP](https://github.com/rtmdrr/testSignificanceNLP)     | statistical significance tests ๋งคํธ๋ฆญ (Normality Check, McNemar, Permutation-randomization, Bootstrap, t-test, Wilcoxon)   |
| [conlleval](https://github.com/sighsmile/conlleval)                      | conlleval in Python (script for chunking/NER evaluation)                          |
| [seqeval](https://github.com/chakki-works/seqeval)                       | A Python framework for sequence labeling evaluation(named-entity recognition, pos tagging, ...)                            |
| [mteval](https://github.com/odashi/mteval)                               | Machine Translation ๊ด๋ จ ๋งคํธ๋ฆญ (BLEU, NIST, RIBES, Word Error Rate)               |


## Model
`๊ด์ฌ์๋ ์ฐ๊ตฌ๋ค ์ ๋ฆฌ`
| Name                                                                          | Description                                                                     |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [Informer2020](https://github.com/zhouhaoyi/Informer2020)                     | AAAI 2021 Best Paper โInformer: Beyond Efficient Transformer for Long Sequence Time-Series Forecastingโ ๋ผ๋ฌธ ๊ตฌํ |
| [parallelformers](https://github.com/tunib-ai/parallelformers)                | Parallelformers: An Efficient Model Parallelization Toolkit for Deployment ๊ตฌํ |
| [KeyBERT](https://github.com/MaartenGr/KeyBERT)                               | KeyBERT ๊ตฌํ                                                                    |
| [ner-bert](https://github.com/sberbank-ai/ner-bert)                           | NER-BERT ๊ตฌํ                                                                   |
| [minGPT](https://github.com/karpathy/minGPT)                                  | OpenAI GPT (Generative Pretrained Transformer) PyTorch ๊ตฌํ                     |
| [minGPT](https://github.com/SeanNaren/minGPT)                                 | OpenAI GPT PyTorch ๊ตฌํ โ DeepSpeed ์ ์ฉ                                        |
| [gpt-j-api](https://github.com/vicgalle/gpt-j-api)                            | API for the GPT-J language model (demo : streamlit/FastAPI)                     |
| [K-BERT](https://github.com/autoliuweijie/K-BERT)                             | AAAI 2020 โK-BERT: Enabling Language Representation with Knowledge Graphโ ๋ผ๋ฌธ ๊ตฌํ |
| [KG-BART](https://github.com/yeliu918/KG-BART)                                | AAAI 2021 โKG-BART: Knowledge Graph-Augmented BART for Generative Commonsense Reasoningโ ๋ผ๋ฌธ ๊ตฌํ |
| [fastT5](https://github.com/Ki6an/fastT5)                                     | fastT5 ๋ชจ๋ธ(T5 ๋ชจ๋ธ ํฌ๊ธฐ 3๋ฐฐ ์ค์ด๊ณ , inference ์๋ 5๋ฐฐ ๋๋ฆผ)                     |
| [PTT5](https://github.com/unicamp-dl/PTT5)                                    | โPTT5: Pretraining and validating the T5 model on Brazilian Portuguese dataโ ๋ผ๋ฌธ ๊ตฌํ                 |
| [kortok](https://github.com/kakaobrain/kortok)                                | AACL-IJCNLP 2020 "An Empirical Study of Tokenization Strategies for Various Korean NLP Tasksโ ๋ผ๋ฌธ ๊ตฌํ     |
| [BertSum](https://github.com/nlpyang/BertSum)                                 | โFine-tune BERT for Extractive Summarizationโ ๋ผ๋ฌธ ๊ตฌํ                         |
| [BERT-Relation-Extraction](https://github.com/plkmo/BERT-Relation-Extraction) | ACL 2019 โMatching the Blanks: Distributional Similarity for Relation Learningโ ๋ผ๋ฌธ ๊ตฌํ                 |
| [keytotext](https://github.com/gagan3012/keytotext)                           | Keywords to Sentences ๋ชจ๋ธ ๊ตฌํ(Model : T5, Demo : Streamlit)                   |
| [mlp-mixer-pytorch](https://github.com/lucidrains/mlp-mixer-pytorch)          | โMLP-Mixer: An all-MLP Architecture for Visionโ ๋ผ๋ฌธ PyTorch ๊ตฌํ               |
| [MLP-Mixer-pytorch](https://github.com/rishikksh20/MLP-Mixer-pytorch)         | โMLP-Mixer: An all-MLP Architecture for Visionโ ๋ผ๋ฌธ PyTorch ๊ตฌํ               |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers)      | EMNLP 2019 โSentence-BERT: Sentence Embeddings using Siamese BERT-Networksโ ๋ผ๋ฌธ ๊ตฌํ                 |
| [Persona-Dialogue-Generation](https://github.com/SivilTaram/Persona-Dialogue-Generation) | ACL2020 "You Impress Me: Dialogue Generation via Mutual Persona Perceptionโ ๋ผ๋ฌธ ๊ตฌํ            |
| [HyperMix](https://github.com/naver-ai/hypermix)                              | EMNLP 2021 โGPT3Mix: Leveraging Large-scale Language Models for Text Augmentationโ ๋ผ๋ฌธ ๊ตฌํ            |
| [minDALL-E](https://github.com/kakaobrain/minDALL-E)                          | PyTorch implementation of a 1.3B text-to-image generation model trained on 14 million image-text pairs   |
| [BERT4doc-Classification](https://github.com/xuyige/BERT4doc-Classification)      | โHow to Fine-Tune BERT for Text Classification?โ ๋ผ๋ฌธ ๊ตฌํ                  |
| [Bart_T5-summarization](https://github.com/renatoviolin/Bart_T5-summarization)    | BART, T5 ์์ฝ ๋ชจ๋ธ (Demo : flask)                                           |
| [Fine_Tuning_T5_for_Summary_Generation](https://github.com/sanazbahargam/Fine_Tuning_T5_for_Summary_Generation) | T5 ์์ฝ ๋ชจ๋ธ                                  |


## Survey
`paper/dataset ๋ฑ ์๋ฒ ์ด ๋ผ๋ฌธ`
| Name                                                                                                | Description                                              |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [Best_AI_paper_2020](https://github.com/louisfb01/Best_AI_paper_2020)                               | 2020๋๋ best paper ๋ชฉ๋ก                                 |
| [EMNLP-2020](https://github.com/juand-r/EMNLP-2020)                                                 | EMNLP 2020 ๋ผ๋ฌธ ๋ชฉ๋ก                                     |
| [EMNLP-2019-Papers](https://github.com/roomylee/EMNLP-2019-Papers)                                  | EMNLP 2019 ๋ผ๋ฌธ ๋ชฉ๋ก                                     |
| [nlp-papers-with-arxiv](https://github.com/roomylee/nlp-papers-with-arxiv)                          | ACL, EMNLP, NACL, EACL, AACL ํต๊ณ ๋ฐ ๋ผ๋ฌธ ๋ชฉ๋ก           |
| [ABigSurvey](https://github.com/NiuTrans/ABigSurvey)                                                | NLP/ML ๊ด๋ จ survey ๋ผ๋ฌธ ๋ชฉ๋ก                             |
| [SOS4NLP](https://github.com/thunlp/SOS4NLP)                                                        | NLP ๊ด๋ จ ๋ผ๋ฌธ ๋ชฉ๋ก                                       |
| [PLMpapers](https://github.com/thunlp/PLMpapers)                                                    | Must-Read Papers on Pre-trained Language Models ๋ชฉ๋ก                                     |
| [ml-surveys](https://github.com/eugeneyan/ml-surveys)                                               | DL, NLP, CV, Graphs, Reinforcement Learning, Recommendations, Graphs ๋ฑ ๋ผ๋ฌธ ๋ชฉ๋ก                      |
| [Awesome-Mobility-Machine-Learning-Contents](https://github.com/zzsza/Awesome-Mobility-Machine-Learning-Contents)                    | ML/DL mobility industry(transportation) ๊ด๋ จ ๋ผ๋ฌธ ๋ฐ ์ปจํ์ธ  ๋ชฉ๋ก |
| [Task-Oriented Dialogue Research Progress Survey](https://github.com/AtmaHou/Task-Oriented-Dialogue-Research-Progress-Survey)        | Task-oriented dialogue ๊ด๋ จ ์๋ฒ ์ด ๋ผ๋ฌธ ๋ฐ ๋ฐ์ดํฐ์ ์ ๋ฆฌ (Dialogue State Tracking, NLU: Slot Filling/Intent Detection)   |
| [Awesome-SLU-Survey](https://github.com/yizhen20133868/Awesome-SLU-Survey)                          | Spoken Language Understanding ๊ด๋ จ ๋ผ๋ฌธ ๋ฐ ๋ฐ์ดํฐ์ ๋ชฉ๋ก        |
| [awesome-speech-recognition-speech-synthesis-papers](https://github.com/zzw922cn/awesome-speech-recognition-speech-synthesis-papers) | Automatic Speech Recognition, Speaker Verification, Voice Conversion, Speech Synthesis, Language Modelling, Confidence Estimates, Music Modelling ๊ด๋ จ ๋ผ๋ฌธ ๋ชฉ๋ก |
| [awesome-nlg](https://github.com/accelerated-text/awesome-nlg)                                      | Natural Language Generation (NLU) ๊ด๋ จ ๋ผ๋ฌธ ๋ฐ ๋ฐ์ดํฐ์ ๋ฑ ์ ๋ฆฌ   |
| [DataAug4NLP](https://github.com/styfeng/DataAug4NLP)                                               | NLP data augmentation ๊ด๋ จ ๋ผ๋ฌธ ๋ชฉ๋ก (paper : A Survey of Data Augmentation Approaches for NLP)       |
| [data-augmentation-review](https://github.com/AgaMiko/data-augmentation-review)                     | Data Augmentation ๊ด๋ จ ๋ผ๋ฌธ ๋ฐ ํจํค์ง ๋ชฉ๋ก               |
| [Dialogue-Generation](https://github.com/csnlp/Dialogue-Generation)                                 | Dialogue Generation ๊ด๋ จ ๋ผ๋ฌธ ๋ชฉ๋ก                       |
| [Question-Generation-Paper-List](https://github.com/teacherpeterpan/Question-Generation-Paper-List) | Neural Question Generation ๊ด๋ จ ๋ผ๋ฌธ ๋ชฉ๋ก                                   |
| [Summarization-Papers](https://github.com/xcfcode/Summarization-Papers)                             | Summarization ๊ด๋ จ ๋ผ๋ฌธ survey ๋ฐ ๋ฐ์ดํฐ์ ๋ชฉ๋ก          |
| [Text-Summarization-Repo](https://github.com/uoneway/Text-Summarization-Repo)                       | Summarization ๊ด๋ จ ๋ผ๋ฌธ ๋ฐ ๋ฐ์ดํฐ์ ๋ชฉ๋ก                 |
| [Awesome-Dialogue-State-Tracking](https://github.com/yukyunglee/Awesome-Dialogue-State-Tracking)    | Dialogue State Tracking (DST) ๊ด๋ จ ๋ผ๋ฌธ ๋ฐ ๋ฐ์ดํฐ์ ๋ชฉ๋ก |
| [BERT-related-papers](https://github.com/tomohideshibata/BERT-related-papers)                       | BERT ๊ด๋ จ ๋ผ๋ฌธ ๋ชฉ๋ก (downstream task ๊น์ง ์ ๋ฆฌ)          |
| [Knowledge-Grounded-Conversation](https://github.com/ChuanMeng/Knowledge-Grounded-Conversation)     | Knowledge Grounded Conversation (KGC) ๊ด๋ จ ๋ผ๋ฌธ ๋ชฉ๋ก    |



## Streamlit
`streamlit ๊ด๋ จ ์ฐธ๊ณ  ์ฝ๋`
| Name                                                                             | Description                                                              |
|----------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [streamlit](https://github.com/streamlit/streamlit)                              | streamlit github                                                         |
| [best-of-streamlit](https://github.com/jrieke/best-of-streamlit)                 | streamlit app์์ best ์์ ๋ชฉ๋ก                                          |
| [awesome-streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit)         | streamlit ๊ด๋ จ ์ ๋ฆฌ                                                       |
| [streamlit_navbar](https://github.com/BugzTheBunny/streamlit_navbar)             | streamlit NavBar ์์                                                      |
| [bertsum-streamlit](https://github.com/gradjitta/bertsum-streamlit)              | summarization streamlit UI ์ฐธ๊ณ                                            |
| [st_ner_annotate](https://github.com/prasadchandan/st_ner_annotate)              | Streamlit Named Entity Recognition (NER) annotation custom component ์ฐธ๊ณ  |
| [streamlit-pandas-profiling](https://github.com/okld/streamlit-pandas-profiling) | Pandas profiling ์ปดํฌ๋ํธ                                                 |



## Subtask
`NLP subtask ๊ตฌํ`
| Name                                                                        | Description                      |
|-----------------------------------------------------------------------------|----------------------------------|
| [nlp_classification](https://github.com/seopbo/nlp_classification)          | NLP Classification ๊ด๋ จ ๋ผ๋ฌธ ๊ตฌํ |
| [BERT-for-Sequence-Labeling-and-Text-Classification](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification) | CoNLL-2003 named entity recognition, Joint Slot Filling and Intent Prediction ํํ๋ฆฟ ์ฝ๋ ์ ๊ณต |
| [KoGPT2-subtasks](https://github.com/haven-jeon/KoGPT2-subtasks)            | subtask ๊ตฌํ(NSMC, KorSTS)        |
| [fine-grained-sentiment](https://github.com/prrao87/fine-grained-sentiment) | SST-5 ๋ฐ์ดํฐ์์ ๋ํด ๋น๊ต         |


## Tutorials
`์ฐธ๊ณ ํ ๋งํ NLP ํํ ๋ฆฌ์ผ`
| Name                                                                                  | Description                                                         |
|---------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [Advanced_Models](https://github.com/jk96491/Advanced_Models)                         | ์ฌ๋ฌ ์ ๊ฒฝ๋ง ๋ชจ๋ธ ๊ตฌํ(DCGAN, CGAN, SA-GAN ,GAN, Resnet, VAE, Multi-Head Attention, GPT-2)             |
| [pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq)                      | PyTorch seq2seq, Attention ๋ชจ๋ธ ๊ตฌํ                                 |
| [transformer-evolution](https://github.com/paul-hyun/transformer-evolution)           | Transformer ์ดํ ๋ชจ๋ธ ๊ตฌํ (Transformer, BERT, ALBERT, SpanBERT, GPT, T5)                           |
| [the-clean-transformer](https://github.com/eubinecto/the-clean-transformer)           | PyTorch-Lightning๊ณผ wandb๋ก ๊ตฌํํ ํธ๋์คํฌ๋จธ                         |
| [nlp-various-tutorials](https://github.com/Huffon/nlp-various-tutorials)              | NLP ๊ด๋ จ ํํ ๋ฆฌ์ผ ์ ๊ณต                                                |
| [nlp-tutorial](https://github.com/graykode/nlp-tutorial)                              | NLP ๊ด๋ จ ํํ ๋ฆฌ์ผ ์ ๊ณต (Basic Embedding Model, CNN, RNN, Attention Mechanism, Model based on Transformer)   |
| [nlp-tasks](https://github.com/sooftware/nlp-tasks)                                   | NLP ๊ด๋ จ ํํ ๋ฆฌ์ผ ์ ๊ณต (Automated Essay Scoring, Automatic Speech Recognition, Dialogue Generation, Dialogue Retrieval, Fill in the Blank, Grammatical Error Correction, Grapheme To Phoneme, Language Modeling, Machine Reading Comprehension, Machine Translation Math Word Problem Solving, Natural Language Inference, Named Entity Recognition, Paraphrase Generation, Phoneme To Grapheme, Sentiment Analysis, Semantic Textual Similarity, Speech Synthesis, Summarization) |
| [NLP-Projects](https://github.com/gaoisbest/NLP-Projects)                             | NLP ๊ด๋ จ ๊ตฌํ (word2vec, sentence2vec, machine reading comprehension, dialog system, text classification, pretrained language model, sequence labeling, information retrieval, information extraction, knowledge graph, text generation, network embedding)                    |
| [nlp_tutorials](https://github.com/seopbo/nlp_tutorials)                              | Huggingface๋ฅผ ์ด์ฉํ downstream task ์ฝ๋ ์ ๊ณต (NSMC, KLUE-YNAT, KLUE-NLI, KLUE-NER, KLUE-MRC, LoRa)    |
| [nlp-notebooks](https://github.com/nlptown/nlp-notebooks)                             | A collection of notebooks for Natural Language Processing from NLP Town  |
| [large-scale-lm-tutorials](https://github.com/tunib-ai/large-scale-lm-tutorials)      | ๋๊ท๋ชจ ์ธ์ด๋ชจ๋ธ ๊ฐ๋ฐ์ ํ์ํ ์ฌ๋ฌ ๊ธฐ์  ์๊ฐ                               |
| [Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials)        | Huggingface์ Transformers ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ํ์ฉํ ํํ ๋ฆฌ์ผ ์ ๊ณต (BERT, DETR, GPT-J-6B, ImageGPT, LayoutLM, T5, Vision Transformer, ...) |
| [EncT5](https://github.com/monologg/EncT5)                       | Pytorch Implementation of EncT5: Fine-tuning T5 Encoder for Non-autoregressive Tasks          |
| [tutorials-kr](https://github.com/PyTorchKorea/tutorials-kr   )                       | PyTorch์์ ์ ๊ณตํ๋ ํํ ๋ฆฌ์ผ ํ๊ธํ                                       |
| [PytorchLightning_TutorialKR](https://github.com/dnap512/PytorchLightning_TutorialKR) | PytorchLightning ํํ ๋ฆฌ์ผ ํ๊ธํ                                          |
| [pytorch-template](https://github.com/victoresque/pytorch-template)                   | ๋ฅ๋ฌ๋ ํ๋ก์ ํธ๋ฅผ ์ํ PyTorch ์ฝ๋ ํํ๋ฆฟ                                 |


## Wiki
`ML/DL/NLP ๊ด๋ จ ์ง์`
| Name                                                                                    | Description                                                             |
|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [๋ชจ๋์ MLOps](https://mlops-for-all.github.io/)                                        | MLOps ๊ด๋ จ ์ ๋ฆฌ                                                          |
| [MLOps-Basics](https://github.com/graviraja/MLOps-Basics)                              | MLOps ๊ด๋ จ ๊ฐ๋ ์ ๋ฆฌ (๋์ด๋ EASY, MEDIUM, HARD๋ก ๊ตฌ๋ถ)                    |
| [100-Days-of-NLP](https://github.com/graviraja/100-Days-of-NLP)                        | NLP ๊ด๋ จ ๊ฐ๋ ์ ๋ฆฌ (๋์ด๋ EASY, MEDIUM, HARD๋ก ๊ตฌ๋ถ)                      |
| [NLUvsNLG](https://github.com/songys/NLUvsNLG)                                         | ์์ฐ์ด์ฒ๋ฆฌ NLU, NLG ์ค๋ช       |
| [awesome-huggingface](https://github.com/huggingface/awesome-huggingface)              | Huggingface ๋ผ์ด๋ธ๋ฌ๋ฆฌ ๋ฐ NLP ๊ด๋ จ ์คํ์์ค ์ ๋ฆฌ                           |
| [nlpbook](https://github.com/ratsgo/nlpbook)                      | ์ค๋ฌด์์ ์ฌ์ฉํ  ์ ์๋ ์์ฐ์ด์ฒ๋ฆฌ ํ ๊ณต์                                                        |
| [Ready-For-Tech-Interview](https://github.com/WooVictory/Ready-For-Tech-Interview)     | ์ ์ ๊ฐ๋ฐ์ ํ์ ์ง์ ์ ๋ฆฌ (Computer Science ์ง์ ์ฐธ๊ณ )                     |
| [tech-interview-for-developer](https://github.com/gyoogle/tech-interview-for-developer)| ์ ์ ๊ฐ๋ฐ์ ์ ๊ณต ์ง์ & ๊ธฐ์  ๋ฉด์  ๋ฐฑ๊ณผ์ฌ์                                    |
| [CSE-Summary](https://github.com/Prev/CSE-Summary)                                     | ์ปดํจํฐ๊ณตํ ์ ๊ณต์ง์ ์ ๋ฆฌ                                                    |
| [ai-tech-interview](https://github.com/boostcamp-ai-tech-4/ai-tech-interview)          | AI ์์ง๋์ด ๊ธฐ์  ๋ฉด์  ์คํฐ๋                                                |
| [machine-learning-interview](https://github.com/khangich/machine-learning-interview)   | ML ์ธํฐ๋ทฐ ์ ๋ฆฌ                                                             |
| [Misc-Cheatsheet](https://github.com/subinium/Misc-Cheatsheet)                         | ๋ํ์ ์ํ์ ํ๋ฉฐ ์ฌ์ฉํ ์ฝ๋ฉํ (Linux, Web, Tool ๋ฑ)                       |
| [Conference-Acceptance-Rate](https://github.com/lixin4ever/Conference-Acceptance-Rate) | AI ๊ด๋ จ ํํ acceptance rate ์ ๋ฆฌ                                           |
| [NLP-conference-compendium](https://github.com/soulbliss/NLP-conference-compendium)    | NLP Top 10 ํํ ์ ๋ฆฌ                                                        |
| [Cite.GG](https://github.com/Beomi/cite.gg)                                            | ๋น์ทํ ๋ผ๋ฌธ ์ถ์ฒ ์ฌ์ดํธ                                                      |
| [arXivNotes](https://github.com/jojonki/arXivNotes)                                    | arXiv ๋ผ๋ฌธ ์ ๋ฆฌ                                                             |
| [Jiphyeonjeon](https://github.com/jiphyeonjeon)                                        | ์งํ์  ๋ผ๋ฌธ ์คํฐ๋ ๋ชฉ๋ก                                                      |
| [ai-seminar](https://github.com/HYU-AILAB/ai-seminar)                                 | ํ์๋ ์ธ๊ณต์ง๋ฅ์ฐ๊ตฌ์ค(AI LAB)์์ ์งํํ๋ ์ธ๋ฏธ๋ ์ ๋ฆฌ                         |
| [Transformer_Survey_Study](https://github.com/yukyunglee/Transformer_Survey_Study)    | ๊ณ ๋ ค๋ DSBA ์ฐ๊ตฌ์ค ํธ๋์คํฌ๋จธ ์ธ๋ฏธ๋ ์ ๋ฆฌ                                     |
| [beyondBERT](https://github.com/modulabs/beyondBERT)                                  | beyondBERT ๋ผ๋ฌธ ์คํฐ๋ ๋ชฉ๋ก                                                  |
| [bookmarks](https://github.com/hyunjun/bookmarks)                                     | ๋ค์ํ ์ง์ ๋ชฉ๋ก (NLP, pytorch, docker ์ฐธ๊ณ )                                 |
| [awesome-python](https://github.com/vinta/awesome-python)                             | ํ์ด์ฌ ๊ด๋ จ ์ง์ ๋ชฉ๋ก                                                        |
| [awesome-react](https://github.com/enaqx/awesome-react)                               | React ๊ด๋ จ ์ง์ ๋ชฉ๋ก                                                         |
| [nlp-startups](https://github.com/Huffon/nlp-startups)                                | ๋ํ๋ฏผ๊ตญ NLP ์คํํธ์ ๋ชฉ๋ก                                                   |
| [Korea-Startups](https://github.com/gyunggyung/Korea-Startups)                        | ๋ํ๋ฏผ๊ตญ ์คํํธ์ ๋ชฉ๋ก (์ฃผ๋ก ์ธ๊ณต์ง๋ฅ ๊ด๋ จ)                                   |
| [gradio](https://github.com/gradio-app/gradio)                                        | ๋จธ์ ๋ฌ๋ ๋ฐ๋ชจ๋ฅผ ์ํ UI ์ฑ (streamlit๊ณผ ์ ์ฌํ๋ input/output ์ปดํฌ๋ํธ์ ์ด์ ) |
| [awesome-demos](https://github.com/gradio-app/awesome-demos)                          | gradio๋ฅผ ํ์ฉํ ๋ฐ๋ชจ ํ์ด์ง ๋ชฉ๋ก                                             |