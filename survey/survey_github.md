# 🗒️ Research 관련 github 정리

DL, NLP 연구 관련 유용한 github을 정리합니다.

*✔️ Last Update : 2022.02.05*

## 🔢 Index
- [🗒️ Research 관련 github 정리](#️-research-관련-github-정리)
  - [🔢 Index](#-index)
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
`크롤러`
| Name                                                                          | Description                                                                     |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [Web_Crawler](https://github.com/BitnaKeum/Web_Crawler)                       | 나무위키, 위키피디아, 다음블로그, 티스토리, 유튜브, 네이트판 크롤러                 |
| [youtube_crawler](https://github.com/Mo0nl19ht/youtube_crawler)               | 유튜브 API를 이용한 키워드에 따른 유튜브 영상 URL, 제목, 상세정보, 댓글, 자막 크롤링 |
| [Youtube_Comment_Crawler](https://github.com/SOMJANG/Youtube_Comment_Crawler) | 유튜브 댓글 크롤러                                                                |
| [naver-blog-crawler](https://github.com/xotrs/naver-blog-crawler)             | 네이버 블로그 포스팅 URL, 포스팅 제목, 포스팅 설명, 포스팅 날짜, 블로거 이름, 포스팅 내용 크롤링 |


## Dataset
`NLP 관련 데이터셋`
| Name                                                                                  | Description                                                       |
|---------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [slot_filling_and_intent_detection_of_SLU](https://github.com/sz128/slot_filling_and_intent_detection_of_SLU) | intent/slot 데이터셋 정리 (ATIS, SNIPS, Facebook’s multilingual dataset, MIT_Restaurant_Movie_corpus, E-commerce Shopping Assistant (ECSA) from Alibaba, CoNLL-2003 NER) |
| [rasa-nlu-benchmark](https://github.com/nghuyong/rasa-nlu-benchmark)                  | Rasa NLU 데이터셋 (ATIS, SNIPS, AskUbuntuCorpus, Facebook Multilingual Task Oriented Dataset, SMP2019, Check flow dataset, MSRA_NER, ToutiaoNews)  |
| [simulated-dialogue](https://github.com/google-research-datasets/simulated-dialogue) | Machines Talking To Machines (M2M) 데이터셋 (Sim-M, Sim-R, Sim-Gen) |
| [NLU-Evaluation-Data](https://github.com/xliuhw/NLU-Evaluation-Data)                 | NLU 데이터셋 (human-robot interaction in home domain)               |
| [nlp-datasets](https://github.com/niderhoff/nlp-datasets)                            | NLP 데이터셋                                                                                                                                                     |
| [nlp-collections](https://github.com/hyunwoongko/nlp-collections)                    | NLP 태스크 데이터셋 목록                                             |
| [paws](https://github.com/google-research-datasets/paws)                             | PAWS: Paraphrase Adversaries from Word Scrambling 데이터셋          |
| [e2e-dataset](https://github.com/tuetschek/e2e-dataset)                              | E2E Challenge 데이터셋                                              |


## Dataset(Ko)
`한국어 데이터셋`
| Name                                                                            | Description                           |
|---------------------------------------------------------------------------------|---------------------------------------|
| [AwesomeKorean_Data](https://github.com/songys/AwesomeKorean_Data)              | 한국어 데이터셋 목록                   |
| [Open-korean-corpora](https://github.com/ko-nlp/Open-korean-corpora)            | 한국어 데이터셋 목록                   |
| [Korpora](https://github.com/ko-nlp/Korpora)                                    | 한국어 데이터셋 오픈소스 파이썬 패키지  |
| [UD_Korean-GSD](https://github.com/UniversalDependencies/UD_Korean-GSD)         | 한국어 의존구문분석 데이터셋            |
| [ud-korean](https://github.com/emorynlp/ud-korean) | 한국어 의존구문분석 데이터셋 |
| [KorNLU Datasets](https://github.com/kakaobrain/KorNLUDatasets)                | KorNLI, KorSTS 데이터셋                 |
| [paraKQC](https://github.com/warnikchow/paraKQC)                               | 한국어 parallel 데이터셋 (10,000 utterances, namely 1,000 sets of 10 similar sentences)   |
| [korean-parallel-corpora](https://github.com/jungyeul/korean-parallel-corpora) | 한국어 parallel 데이터셋                |
| [korean-hate-speech](https://github.com/kocohub/korean-hate-speech)            | 한국어 혐오 데이터셋                    |
| [nsmc](https://github.com/e9t/nsmc)                                            | 네이버 영화리뷰 감성분석 데이터셋 (Naver sentiment movie corpus)  |
| [KorAdvMRSTestData](https://github.com/kakaoenterprise/KorAdvMRSTestData)      | 한국어 오픈 도메인 대화 답변 선택 모델의 취약점을 평가하기 위한 데이터셋 EMNLP 2021 “An Evaluation Dataset and Strategy for Building Robust Multi-turn Response Selection Model” 논문 |


## Generation
`Text Generation/Data Augmentation 관련 연구 정리`
| Name                                                                                    | Description                                                    |
|-----------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [nlpaug](https://github.com/makcedward/nlpaug)                                          | NLP 관련 augmentation 패키지                                    |
| [contextual_augmentation](https://github.com/pfnet-research/contextual_augmentation)    | NAACL 2018 “Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations” 논문 구현 |
| [Seq2SeqDataAugmentationForLU](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) | COLING 2018 “Sequence-to-sequence Data Augmentation for Dialogue Language Understanding” 논문 구현          |
| [Texygen](https://github.com/geek-ai/Texygen)                                           | SIGIR 2018 “Texygen: A Benchmarking Platform for Text Generation Models” 논문 구현                         |
| [eda_nlp](https://github.com/jasonwei20/eda_nlp)                                        | MNLP-IJCNLP 2019 “EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks“ 논문 구현 |
| [cbert_aug](https://github.com/1024er/cbert_aug)                                        | ICCS 2019 "Conditional BERT Contextual Augmentation” 논문 구현  |
| [TransformersDataAugmentation](https://github.com/varunkumar-dev/TransformersDataAugmentation) | AACL 2020 “Data Augmentation Using Pre-trained Transformer Models” 논문 구현 (Basline : EDA, Backtranslation, CBERT)   |
| [GPT-GNN](https://github.com/acbull/GPT-GNN)                                            | KDD 2020 “GPT-GNN: Generative Pre-Training of Graph Neural Networks” 논문 구현                              |
| [PPLM](https://github.com/uber-research/PPLM)                                           | ICLR 2020 “Plug and Play Language Models: A Simple Approach to Controlled Text Generation” 논문 구현        |
| [SentAugment](https://github.com/facebookresearch/SentAugment)                          | NAACL 2021 “Self-training Improves Pre-training for Natural Language Understanding” 논문 구현          |
| [text-autoaugment](https://github.com/lancopku/text-autoaugment)                        | EMNLP 2021 “Text AutoAugment: Learning Compositional Augmentation Policy for Text Classification” 논문 구현    |
| [Paraphrase-Generator](https://github.com/Vamsi995/Paraphrase-Generator)                | Paraphrase Generator 모델 (dataset : Google’s PAWS, model : T5, demo : streamlit/flask)                 |
| [T5-paraphrase-generation](https://github.com/renatoviolin/T5-paraphrase-generation)    | Paraphrase Generator 모델 (model : T5, demo : flask)  |
| [ctrl-sum](https://github.com/salesforce/ctrl-sum)                                      | ICLR 2021 “CTRLsum: Towards Generic Controllable Text Summarization” 논문 구현                             |
| [question_generation](https://github.com/patil-suraj/question_generation)               | Neural question generation using transformers    |
| [textaugment](https://github.com/dsfsi/textaugment)                                     | Text Augmentation 라이브러리 (paper : “Improving Short Text Classification Through Global Augmentation Methods”) |



## Infra
`딥러닝 환경 구축`
| Name                                                             | Description                                                      |
|------------------------------------------------------------------|------------------------------------------------------------------|
| [gpustat](https://github.com/wookayin/gpustat)                   | GPU 모니터링 툴                                                   |
| [deepo](https://github.com/ufoym/deepo)                          | 딥러닝 환경 구축을 위한 도커 이미지 제공                           |
| [_Book_k8sInfra](https://github.com/sysnet4admin/_Book_k8sInfra) | “컨테이너 인프라 환경 구축을 위한 쿠버네티스/도커” 책 실습 코드 제공 |



## Korean NLP
`한국어 NLP 프로젝트`
| Name                                                                                                | Description                                                 |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [KoEDA](https://github.com/toriving/KoEDA)                                                          | 한국어 EDA (Easy Data Augmentation) 모델                     |
| [KorEDA](https://github.com/catSirup/KorEDA)                                                        | 한국어 EDA 모델                                              |
| [Sentence_BERT_Korean](https://github.com/robinsongh381/Sentence_BERT_Korean)                       | Sentence-BERT 한국어 모델 구현 (dataset : KorNLI, model : DistilKoBERT) |
| [KoSentenceBERT-ETRI](https://github.com/BM-K/KoSentenceBERT-ETRI)                                  | Sentence-BERT 한국어 모델 구현 (dataset : KorNLU, model : ETRI KorBERT)      |
| [KoSentenceBERT-SKT](https://github.com/BM-K/KoSentenceBERT-SKT)                                    | Sentence-BERT 한국어 모델 구현 (dataset : KorNLU, model : SKT KoBERT)       |
| [NLP_Koeran_DP](https://github.com/hanjanghoon/NLP_Koeran_DP)                                       | 한국어 의존구문분석 모델 개발 (2019 국어경진대회 한국어 의존구문 분석 대상)     |
| [PyKoSpacing](https://github.com/haven-jeon/PyKoSpacing)                                            | 한국어 띄어쓰기 모델                                          |
| [KoBART-chatbot](https://github.com/haven-jeon/KoBART-chatbot)                                      | 한국어 KoBART 챗봇 모델                                       |
| [KoBART-summarization](https://github.com/seujung/KoBART-summarization)                             | 한국어 KoBART 요약 모델                                       |
| [t5-summarization](https://github.com/seujung/t5-summarization)                                     | 한국어 T5 요약 모델                                           |
| [KoGPT2-DINO](https://github.com/soeque1/KoGPT2-DINO)                                               | 한국어 DINO(Generating Datasets with Pretrained Language Models 논문) 모델    |
| [KoSimCSE-SKT](https://github.com/BM-K/KoSimCSE-SKT)                                                | 한국어 SimCSE 모델 (paper : SimCSE: Simple Contrastive Learning of Sentence Embeddings) (dataset : KorNLI, model : SKT KoBERT) |
| [WellnessConversation-LanguageModel](https://github.com/nawnoes/WellnessConversation-LanguageModel) | AI 허브 정신건강 상담 데이터를 활용한 심리상담 대화 모델 (KoBERT, KoELECTRA, KoGPT2)  |
| [klue-transformers-tutorial](https://github.com/Huffon/klue-transformers-tutorial)                  | KLUE 데이터를 활용한 HuggingFace Transformers 튜토리얼 (NLI, STS, zero shot classification) |
| [Awesome-Korean-NLP-Papers](https://github.com/changukshin/Awesome-Korean-NLP-Papers)              | 한국어 NLP 논문 목록 (~2019년도까지 한국학회 위주로 정리)        |
| [hama-py](https://github.com/hamanlp/hama-py)                                                      | 파이썬 한글 처리 라이브러리 (형태소 분석, 품사 태깅)                                                                           |
| [hangul-toolkit](https://github.com/bluedisk/hangul-toolkit)                                       | 한글 자모 분리/조합 작업을 위한 툴킷                            |
| [kss](https://github.com/hyunwoongko/kss)                                                          | 한국어 문장 분리 라이브러리 (Kss: A Toolkit for Korean sentence segmentation)                                                  |
| [korean-sentence-splitter](https://github.com/likejazz/korean-sentence-splitter)                   | 한국어 문장 분리 라이브러리                                     |
| [ko_lm_dataformat](https://github.com/monologg/ko_lm_dataformat)                                   | 한국어 언어모델용 학습 데이터를 저장, 로딩하기 위한 유틸리티 (zstandard, ultrajson 을 사용하여 데이터 로딩 및 압축 속도 개선)  |
| [kospeech](https://github.com/sooftware/kospeech)                                                  | 한국어 음성 인식 툴킷 (paper : Open-source toolkit for end-to-end Korean speech recognition)  |
| [Awesome-Korean-NLP](https://github.com/datanada/Awesome-Korean-NLP)                               | 한국어 NLP 관련 정리                                           |
| [awesome-hangul](https://github.com/lqez/awesome-hangul)                                           | 한글/한국어 처리 관련 라이브러리 및 모듈 목록                    |


## Latex / Notion / README
`Latex, Notion, README에 사용하기 좋은 템플릿 정리`
| Name                                                                                                    | Description                                             |
|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| [Notion-to-GitHub-Pages](https://github.com/uoneway/notion-to-github-pages)                             | 노션 페이지를 Github Pages 블로그에 맞게 업로드하는 shell script |
| [notion-py](https://github.com/jamalex/notion-py)                                                       | 비공식 notion API                                       |
| [Best-README-Template](https://github.com/othneildrew/Best-README-Template)                             | README 템플릿 정리                                      |
| [README-template.md ](https://github.com/scottydocs/README-template.md)                                 | README 템플릿 (Prerequisites, Installing, Using 관련)   |
| [markdown-badges](https://github.com/Ileriayo/markdown-badges)                                          | markdown 뱃지 목록                                      |
| [github-readme-stats](https://github.com/anuraghazra/github-readme-stats/blob/master/docs/readme_kr.md) | readme stats 표시                                       |
| [productive-box](https://github.com/maxam2017/productive-box)                                           | commit 시각 통계 노출                                    |
| [awesome-pinned-gists](https://github.com/matchai/awesome-pinned-gists)                                 | A collection of awesome dynamic pinned gists for GitHub |
| [Awesome-CV](https://github.com/posquit0/Awesome-CV)                                                    | CV Latex 템플릿                                         |
| [machine-learning-cheat-sheet](https://github.com/soulmachine/machine-learning-cheat-sheet)             | machine learning cheat sheet(ML 개념 정리) Latex 템플릿  |


## Libraries / Toolkits
`활용하기 좋은 패키지/툴킷`
| Name                                                                                | Description                                                             |
|--------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [lightning-transformers](https://github.com/PyTorchLightning/lightning-transformers) | Transformers SOTA 모델 가져다 쓸 수 있는 패키지                          |
| [happy-transformer](https://github.com/EricFillion/happy-transformer)                | SOTA NLP 모델 활용할 수 있는 패키지 (Text Generation, Text Classification, Word Prediction, Question Answering, Text-to-Text, Next Sentence Prediction, Token Classification) |
| [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)         | Transformers 모델 train/evaluate 할 수 있는 패키지                       |
| [haystack](https://github.com/deepset-ai/haystack)                                   | Transformer 모델 활용할 수 있는 패키지 (Neural Search, Question Answering, Semantic Document Search, Summarization)   |
| [flair](https://github.com/flairNLP/flair)                                           | SOTA NLP 모델 활용할 수 있는 패키지                                      |
| [autonlp](https://github.com/huggingface/autonlp)                                    | SOTA NLP 모델 활용할 수 있는 패키지                                      |
| [notebooks](https://github.com/huggingface/notebooks)                                | Huggingface 라이브러리 notebooks                                        |
| [NeMo](https://github.com/NVIDIA/NeMo)                                               | conversational AI 툴킷 (Automatic Speech Recognition, Natural Language Processing, and Text-to-Speech Synthesis)  |
| [texar-pytorch](https://github.com/asyml/texar-pytorch)                              | Machine Learning, Natural Language Processing, Text Generation 지원하기 위한 툴킷, ML 모듈 및 라이브러리 제공             |
| [claf](https://github.com/naver/claf)                                               | CLaF: Open-Source Clova Language Framework                               |
| [graph4nlp](https://github.com/graph4ai/graph4nlp)                                  | NLP에서의 Graph Neural Networks 쉽게 사용하기 위한 패키지                                      |
| [transformers-interpret](https://github.com/cdpierse/transformers-interpret)        | Huggingface Transformers 라이브러리를 활용하여 쉽게 모델을 설명할 수 있는 패키지 (Sequence Classification, Zero Shot Classification, Question Answering)      |
| [self-attentive-parser](https://github.com/nikitakit/self-attentive-parser)         | 11개 언어에 대한 constituency parsing 패키지 (SpaCy benepar Berkeley Neural Parser)                                     |
| [yanmtt](https://github.com/prajdabre/yanmtt)                                       | Yet Another Neural Machine Translation 툴킷                                        |
| [OpenPrompt](https://github.com/thunlp/OpenPrompt)                                  | An Open-Source Framework for Prompt-Learning                           |
| [oslo](https://github.com/tunib-ai/oslo)                                            | Open Source framework for Large-scale transformer Optimization                                |
| [bertviz](https://github.com/jessevig/bertviz)                                      | 트랜스포머 모델 시각화 ACL 2019 “A Multiscale Visualization of Attention in the Transformer Model” 논문 구현                 |
| [summvis](https://github.com/robustness-gym/summvis)                                | Summarization 분석 툴 ACl 2021 "SummVis: Interactive Visual Analysis of Models, Data, and Evaluation for Text Summarization” 논문 구현 |
| [lassl](https://github.com/lassl/lassl)                                             | Easy Language Model Pretraining leveraging Huggingface's Transformers and Datasets                                    |
| [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) | PyTorch metric learning 구현 (TripletMarginLoss, ...)                   |
| [pytorch-summary](https://github.com/sksq96/pytorch-summary)                        | Keras의 model.summary()와 유사한 함수 PyTorch로 구현                     |
| [iterative-stratification](https://github.com/trent-b/iterative-stratification)     | scikit-learn cross validators (MultilabelStratifiedKFold, RepeatedMultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit) |



## Metric
`NLP metric`
| Name                                                                     | Description                                                                     |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [NLPMetrics](https://github.com/gcunhase/NLPMetrics)                     | NLP 평가 매트릭(BLEU, GLEU, WER, METEOR, TER, ROUGE, CIDEr)                      |
| [nlg-eval](https://github.com/Maluuba/nlg-eval)                          | Natural Language Generation 자동 평가 매트릭 (BLEU, METEOR, ROUGE, CIDEr, SkipThought cosine similarity, Embedding Average cosine similarity, Vector Extrema cosine similarity, Greedy Matching score) |
| [glge](https://github.com/microsoft/glge)                                | ACL 2021 “GLGE: A New General Language Generation Evaluation Benchmark” 논문 구현 |
| [dialoglue](https://github.com/alexa/dialoglue)                          | DialoGLUE: A Natural Language Understanding Benchmark for Task-Oriented Dialogue  |
| [bert_score](https://github.com/Tiiiger/bert_score)                      | ICLR 2020 “BERTScore: Evaluating Text Generation with BERT” 논문 구현             |
| [BARTScore](https://github.com/neulab/BARTScore)                         | NIPS 2021 “BARTScore: Evaluating Generated Text as Text Generation” 논문 구현     |
| [sacrebleu](https://github.com/mjpost/sacrebleu)                         | Proceedings of the Third Conference on Machine Translation 2018 “A Call for Clarity in Reporting BLEU Scores” 논문 구현 |
| [mlm-scoring](https://github.com/awslabs/mlm-scoring)                    | ACL 2020 “Masked Language Model Scoring” 논문 구현                                |
| [sentence-similarity](https://github.com/tuzhucheng/sentence-similarity) | Paraphrase Detection, Semantic Texual Similarity, Natural Language Inference / Textual Entailment, Answer Selection 관련 매트릭 |
| [testSignificanceNLP](https://github.com/rtmdrr/testSignificanceNLP)     | statistical significance tests 매트릭 (Normality Check, McNemar, Permutation-randomization, Bootstrap, t-test, Wilcoxon)   |
| [conlleval](https://github.com/sighsmile/conlleval)                      | conlleval in Python (script for chunking/NER evaluation)                          |
| [seqeval](https://github.com/chakki-works/seqeval)                       | A Python framework for sequence labeling evaluation(named-entity recognition, pos tagging, ...)                            |
| [mteval](https://github.com/odashi/mteval)                               | Machine Translation 관련 매트릭 (BLEU, NIST, RIBES, Word Error Rate)               |


## Model
`관심있는 연구들 정리`
| Name                                                                          | Description                                                                     |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [Informer2020](https://github.com/zhouhaoyi/Informer2020)                     | AAAI 2021 Best Paper “Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting” 논문 구현 |
| [parallelformers](https://github.com/tunib-ai/parallelformers)                | Parallelformers: An Efficient Model Parallelization Toolkit for Deployment 구현 |
| [KeyBERT](https://github.com/MaartenGr/KeyBERT)                               | KeyBERT 구현                                                                    |
| [ner-bert](https://github.com/sberbank-ai/ner-bert)                           | NER-BERT 구현                                                                   |
| [minGPT](https://github.com/karpathy/minGPT)                                  | OpenAI GPT (Generative Pretrained Transformer) PyTorch 구현                     |
| [minGPT](https://github.com/SeanNaren/minGPT)                                 | OpenAI GPT PyTorch 구현 → DeepSpeed 적용                                        |
| [gpt-j-api](https://github.com/vicgalle/gpt-j-api)                            | API for the GPT-J language model (demo : streamlit/FastAPI)                     |
| [K-BERT](https://github.com/autoliuweijie/K-BERT)                             | AAAI 2020 “K-BERT: Enabling Language Representation with Knowledge Graph” 논문 구현 |
| [KG-BART](https://github.com/yeliu918/KG-BART)                                | AAAI 2021 “KG-BART: Knowledge Graph-Augmented BART for Generative Commonsense Reasoning” 논문 구현 |
| [fastT5](https://github.com/Ki6an/fastT5)                                     | fastT5 모델(T5 모델 크기 3배 줄이고, inference 속도 5배 늘림)                     |
| [PTT5](https://github.com/unicamp-dl/PTT5)                                    | “PTT5: Pretraining and validating the T5 model on Brazilian Portuguese data” 논문 구현                 |
| [kortok](https://github.com/kakaobrain/kortok)                                | AACL-IJCNLP 2020 "An Empirical Study of Tokenization Strategies for Various Korean NLP Tasks” 논문 구현     |
| [BertSum](https://github.com/nlpyang/BertSum)                                 | “Fine-tune BERT for Extractive Summarization” 논문 구현                         |
| [BERT-Relation-Extraction](https://github.com/plkmo/BERT-Relation-Extraction) | ACL 2019 “Matching the Blanks: Distributional Similarity for Relation Learning” 논문 구현                 |
| [keytotext](https://github.com/gagan3012/keytotext)                           | Keywords to Sentences 모델 구현(Model : T5, Demo : Streamlit)                   |
| [mlp-mixer-pytorch](https://github.com/lucidrains/mlp-mixer-pytorch)          | “MLP-Mixer: An all-MLP Architecture for Vision” 논문 PyTorch 구현               |
| [MLP-Mixer-pytorch](https://github.com/rishikksh20/MLP-Mixer-pytorch)         | “MLP-Mixer: An all-MLP Architecture for Vision” 논문 PyTorch 구현               |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers)      | EMNLP 2019 “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks” 논문 구현                 |
| [Persona-Dialogue-Generation](https://github.com/SivilTaram/Persona-Dialogue-Generation) | ACL2020 "You Impress Me: Dialogue Generation via Mutual Persona Perception” 논문 구현            |
| [HyperMix](https://github.com/naver-ai/hypermix)                              | EMNLP 2021 “GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation” 논문 구현            |
| [minDALL-E](https://github.com/kakaobrain/minDALL-E)                          | PyTorch implementation of a 1.3B text-to-image generation model trained on 14 million image-text pairs   |
| [BERT4doc-Classification](https://github.com/xuyige/BERT4doc-Classification)      | “How to Fine-Tune BERT for Text Classification?” 논문 구현                  |
| [Bart_T5-summarization](https://github.com/renatoviolin/Bart_T5-summarization)    | BART, T5 요약 모델 (Demo : flask)                                           |
| [Fine_Tuning_T5_for_Summary_Generation](https://github.com/sanazbahargam/Fine_Tuning_T5_for_Summary_Generation) | T5 요약 모델                                  |


## Survey
`paper/dataset 등 서베이 논문`
| Name                                                                                                | Description                                              |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [Best_AI_paper_2020](https://github.com/louisfb01/Best_AI_paper_2020)                               | 2020년도 best paper 목록                                 |
| [EMNLP-2020](https://github.com/juand-r/EMNLP-2020)                                                 | EMNLP 2020 논문 목록                                     |
| [EMNLP-2019-Papers](https://github.com/roomylee/EMNLP-2019-Papers)                                  | EMNLP 2019 논문 목록                                     |
| [nlp-papers-with-arxiv](https://github.com/roomylee/nlp-papers-with-arxiv)                          | ACL, EMNLP, NACL, EACL, AACL 통계 및 논문 목록           |
| [ABigSurvey](https://github.com/NiuTrans/ABigSurvey)                                                | NLP/ML 관련 survey 논문 목록                             |
| [SOS4NLP](https://github.com/thunlp/SOS4NLP)                                                        | NLP 관련 논문 목록                                       |
| [PLMpapers](https://github.com/thunlp/PLMpapers)                                                    | Must-Read Papers on Pre-trained Language Models 목록                                     |
| [ml-surveys](https://github.com/eugeneyan/ml-surveys)                                               | DL, NLP, CV, Graphs, Reinforcement Learning, Recommendations, Graphs 등 논문 목록                      |
| [Awesome-Mobility-Machine-Learning-Contents](https://github.com/zzsza/Awesome-Mobility-Machine-Learning-Contents)                    | ML/DL mobility industry(transportation) 관련 논문 및 컨텐츠 목록 |
| [Task-Oriented Dialogue Research Progress Survey](https://github.com/AtmaHou/Task-Oriented-Dialogue-Research-Progress-Survey)        | Task-oriented dialogue 관련 서베이 논문 및 데이터셋 정리 (Dialogue State Tracking, NLU: Slot Filling/Intent Detection)   |
| [Awesome-SLU-Survey](https://github.com/yizhen20133868/Awesome-SLU-Survey)                          | Spoken Language Understanding 관련 논문 및 데이터셋 목록        |
| [awesome-speech-recognition-speech-synthesis-papers](https://github.com/zzw922cn/awesome-speech-recognition-speech-synthesis-papers) | Automatic Speech Recognition, Speaker Verification, Voice Conversion, Speech Synthesis, Language Modelling, Confidence Estimates, Music Modelling 관련 논문 목록 |
| [awesome-nlg](https://github.com/accelerated-text/awesome-nlg)                                      | Natural Language Generation (NLU) 관련 논문 및 데이터셋 등 정리   |
| [DataAug4NLP](https://github.com/styfeng/DataAug4NLP)                                               | NLP data augmentation 관련 논문 목록 (paper : A Survey of Data Augmentation Approaches for NLP)       |
| [data-augmentation-review](https://github.com/AgaMiko/data-augmentation-review)                     | Data Augmentation 관련 논문 및 패키지 목록               |
| [Dialogue-Generation](https://github.com/csnlp/Dialogue-Generation)                                 | Dialogue Generation 관련 논문 목록                       |
| [Question-Generation-Paper-List](https://github.com/teacherpeterpan/Question-Generation-Paper-List) | Neural Question Generation 관련 논문 목록                                   |
| [Summarization-Papers](https://github.com/xcfcode/Summarization-Papers)                             | Summarization 관련 논문 survey 및 데이터셋 목록          |
| [Text-Summarization-Repo](https://github.com/uoneway/Text-Summarization-Repo)                       | Summarization 관련 논문 및 데이터셋 목록                 |
| [Awesome-Dialogue-State-Tracking](https://github.com/yukyunglee/Awesome-Dialogue-State-Tracking)    | Dialogue State Tracking (DST) 관련 논문 및 데이터셋 목록 |
| [BERT-related-papers](https://github.com/tomohideshibata/BERT-related-papers)                       | BERT 관련 논문 목록 (downstream task 까지 정리)          |
| [Knowledge-Grounded-Conversation](https://github.com/ChuanMeng/Knowledge-Grounded-Conversation)     | Knowledge Grounded Conversation (KGC) 관련 논문 목록    |



## Streamlit
`streamlit 관련 참고 코드`
| Name                                                                             | Description                                                              |
|----------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [streamlit](https://github.com/streamlit/streamlit)                              | streamlit github                                                         |
| [best-of-streamlit](https://github.com/jrieke/best-of-streamlit)                 | streamlit app에서 best 순위 목록                                          |
| [awesome-streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit)         | streamlit 관련 정리                                                       |
| [streamlit_navbar](https://github.com/BugzTheBunny/streamlit_navbar)             | streamlit NavBar 예제                                                     |
| [bertsum-streamlit](https://github.com/gradjitta/bertsum-streamlit)              | summarization streamlit UI 참고                                           |
| [st_ner_annotate](https://github.com/prasadchandan/st_ner_annotate)              | Streamlit Named Entity Recognition (NER) annotation custom component 참고 |
| [streamlit-pandas-profiling](https://github.com/okld/streamlit-pandas-profiling) | Pandas profiling 컴포넌트                                                 |



## Subtask
`NLP subtask 구현`
| Name                                                                        | Description                      |
|-----------------------------------------------------------------------------|----------------------------------|
| [nlp_classification](https://github.com/seopbo/nlp_classification)          | NLP Classification 관련 논문 구현 |
| [BERT-for-Sequence-Labeling-and-Text-Classification](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification) | CoNLL-2003 named entity recognition, Joint Slot Filling and Intent Prediction 템플릿 코드 제공 |
| [KoGPT2-subtasks](https://github.com/haven-jeon/KoGPT2-subtasks)            | subtask 구현(NSMC, KorSTS)        |
| [fine-grained-sentiment](https://github.com/prrao87/fine-grained-sentiment) | SST-5 데이터셋에 대해 비교         |


## Tutorials
`참고할만한 NLP 튜토리얼`
| Name                                                                                  | Description                                                         |
|---------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [Advanced_Models](https://github.com/jk96491/Advanced_Models)                         | 여러 신경망 모델 구현(DCGAN, CGAN, SA-GAN ,GAN, Resnet, VAE, Multi-Head Attention, GPT-2)             |
| [pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq)                      | PyTorch seq2seq, Attention 모델 구현                                 |
| [transformer-evolution](https://github.com/paul-hyun/transformer-evolution)           | Transformer 이후 모델 구현 (Transformer, BERT, ALBERT, SpanBERT, GPT, T5)                           |
| [the-clean-transformer](https://github.com/eubinecto/the-clean-transformer)           | PyTorch-Lightning과 wandb로 구현한 트랜스포머                         |
| [nlp-various-tutorials](https://github.com/Huffon/nlp-various-tutorials)              | NLP 관련 튜토리얼 제공                                                |
| [nlp-tutorial](https://github.com/graykode/nlp-tutorial)                              | NLP 관련 튜토리얼 제공 (Basic Embedding Model, CNN, RNN, Attention Mechanism, Model based on Transformer)   |
| [nlp-tasks](https://github.com/sooftware/nlp-tasks)                                   | NLP 관련 튜토리얼 제공 (Automated Essay Scoring, Automatic Speech Recognition, Dialogue Generation, Dialogue Retrieval, Fill in the Blank, Grammatical Error Correction, Grapheme To Phoneme, Language Modeling, Machine Reading Comprehension, Machine Translation Math Word Problem Solving, Natural Language Inference, Named Entity Recognition, Paraphrase Generation, Phoneme To Grapheme, Sentiment Analysis, Semantic Textual Similarity, Speech Synthesis, Summarization) |
| [NLP-Projects](https://github.com/gaoisbest/NLP-Projects)                             | NLP 관련 구현 (word2vec, sentence2vec, machine reading comprehension, dialog system, text classification, pretrained language model, sequence labeling, information retrieval, information extraction, knowledge graph, text generation, network embedding)                    |
| [nlp_tutorials](https://github.com/seopbo/nlp_tutorials)                              | Huggingface를 이용한 downstream task 코드 제공 (NSMC, KLUE-YNAT, KLUE-NLI, KLUE-NER, KLUE-MRC, LoRa)    |
| [nlp-notebooks](https://github.com/nlptown/nlp-notebooks)                             | A collection of notebooks for Natural Language Processing from NLP Town  |
| [large-scale-lm-tutorials](https://github.com/tunib-ai/large-scale-lm-tutorials)      | 대규모 언어모델 개발에 필요한 여러 기술 소개                               |
| [Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials)        | Huggingface의 Transformers 라이브러리를 활용한 튜토리얼 제공 (BERT, DETR, GPT-J-6B, ImageGPT, LayoutLM, T5, Vision Transformer, ...) |
| [EncT5](https://github.com/monologg/EncT5)                       | Pytorch Implementation of EncT5: Fine-tuning T5 Encoder for Non-autoregressive Tasks          |
| [tutorials-kr](https://github.com/PyTorchKorea/tutorials-kr   )                       | PyTorch에서 제공하는 튜토리얼 한글화                                       |
| [PytorchLightning_TutorialKR](https://github.com/dnap512/PytorchLightning_TutorialKR) | PytorchLightning 튜토리얼 한글화                                          |
| [pytorch-template](https://github.com/victoresque/pytorch-template)                   | 딥러닝 프로젝트를 위한 PyTorch 코드 템플릿                                 |


## Wiki
`ML/DL/NLP 관련 지식`
| Name                                                                                    | Description                                                             |
|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [모두의 MLOps](https://mlops-for-all.github.io/)                                        | MLOps 관련 정리                                                          |
| [MLOps-Basics](https://github.com/graviraja/MLOps-Basics)                              | MLOps 관련 개념 정리 (난이도 EASY, MEDIUM, HARD로 구분)                    |
| [100-Days-of-NLP](https://github.com/graviraja/100-Days-of-NLP)                        | NLP 관련 개념 정리 (난이도 EASY, MEDIUM, HARD로 구분)                      |
| [NLUvsNLG](https://github.com/songys/NLUvsNLG)                                         | 자연어처리 NLU, NLG 설명       |
| [awesome-huggingface](https://github.com/huggingface/awesome-huggingface)              | Huggingface 라이브러리 및 NLP 관련 오픈소스 정리                           |
| [nlpbook](https://github.com/ratsgo/nlpbook)                      | 실무에서 사용할 수 있는 자연어처리 팁 공유                                                       |
| [Ready-For-Tech-Interview](https://github.com/WooVictory/Ready-For-Tech-Interview)     | 신입 개발자 필요 지식 정리 (Computer Science 지식 참고)                     |
| [tech-interview-for-developer](https://github.com/gyoogle/tech-interview-for-developer)| 신입 개발자 전공 지식 & 기술 면접 백과사전                                   |
| [CSE-Summary](https://github.com/Prev/CSE-Summary)                                     | 컴퓨터공학 전공지식 정리                                                    |
| [ai-tech-interview](https://github.com/boostcamp-ai-tech-4/ai-tech-interview)          | AI 엔지니어 기술 면접 스터디                                                |
| [machine-learning-interview](https://github.com/khangich/machine-learning-interview)   | ML 인터뷰 정리                                                             |
| [Misc-Cheatsheet](https://github.com/subinium/Misc-Cheatsheet)                         | 대학원 생활을 하며 사용한 코딩팁 (Linux, Web, Tool 등)                       |
| [Conference-Acceptance-Rate](https://github.com/lixin4ever/Conference-Acceptance-Rate) | AI 관련 학회 acceptance rate 정리                                           |
| [NLP-conference-compendium](https://github.com/soulbliss/NLP-conference-compendium)    | NLP Top 10 학회 정리                                                        |
| [Cite.GG](https://github.com/Beomi/cite.gg)                                            | 비슷한 논문 추천 사이트                                                      |
| [arXivNotes](https://github.com/jojonki/arXivNotes)                                    | arXiv 논문 정리                                                             |
| [Jiphyeonjeon](https://github.com/jiphyeonjeon)                                        | 집현전 논문 스터디 목록                                                      |
| [ai-seminar](https://github.com/HYU-AILAB/ai-seminar)                                 | 한양대 인공지능연구실(AI LAB)에서 진행하는 세미나 정리                         |
| [Transformer_Survey_Study](https://github.com/yukyunglee/Transformer_Survey_Study)    | 고려대 DSBA 연구실 트랜스포머 세미나 정리                                     |
| [beyondBERT](https://github.com/modulabs/beyondBERT)                                  | beyondBERT 논문 스터디 목록                                                  |
| [bookmarks](https://github.com/hyunjun/bookmarks)                                     | 다양한 지식 목록 (NLP, pytorch, docker 참고)                                 |
| [awesome-python](https://github.com/vinta/awesome-python)                             | 파이썬 관련 지식 목록                                                        |
| [awesome-react](https://github.com/enaqx/awesome-react)                               | React 관련 지식 목록                                                         |
| [nlp-startups](https://github.com/Huffon/nlp-startups)                                | 대한민국 NLP 스타트업 목록                                                   |
| [Korea-Startups](https://github.com/gyunggyung/Korea-Startups)                        | 대한민국 스타트업 목록 (주로 인공지능 관련)                                   |
| [gradio](https://github.com/gradio-app/gradio)                                        | 머신러닝 데모를 위한 UI 앱 (streamlit과 유사하나 input/output 컴포넌트에 초점) |
| [awesome-demos](https://github.com/gradio-app/awesome-demos)                          | gradio를 활용한 데모 페이지 목록                                             |