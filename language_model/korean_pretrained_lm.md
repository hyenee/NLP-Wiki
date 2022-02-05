# 🗒️ 한국어 선학습 언어모델(Korean Pre-trained Language Model)

*✔️ Last Update : 2022.02.05*


## 🔢 Index
- [🗒️ 한국어 선학습 언어모델(Korean Pre-trained Language Model)](#️-한국어-선학습-언어모델korean-pre-trained-language-model)
  - [🔢 Index](#-index)
  - [1️⃣ Transformer Encoder 기반](#1️⃣-transformer-encoder-기반)
  - [2️⃣ Transformer Decoder 기반](#2️⃣-transformer-decoder-기반)
  - [3️⃣ Transformer Encoder-Decoder 기반](#3️⃣-transformer-encoder-decoder-기반)
  - [4️⃣ 기타](#4️⃣-기타)


## 1️⃣ Transformer Encoder 기반

| Model | Description | Dataset | Vocab Size | Tokenizer |
|-------|-------------|---------|------------|-----------|
| [BERT multilingual (Google)](https://github.com/google-research/bert) | | 위키피디아 (100개 언어 이상) | 119,547 | WordPiece |
| [KoBERT (SKTBrain)](https://github.com/SKTBrain/KoBERT) | SKTBrain에서 배포한 한국어 BERT 모델 | 한국어위키 (문장 5M, 단어 54M) | 8,002 | Sentencepiece |
| [KorBERT (ETRI)](https://aiopen.etri.re.kr/service_dataset.php) | ETRI 엑소브레인 연구진이 배포하는 한국어 BERT 모델로 Korean_BERT_Morphology(형태소분석 기반), Korean_BERT_WordPiece(어절 기반) 모델 제공 | 신문기사와 백과사전 등 23GB (47억개 형태소) | morphology : 30,349 / wordpiece : 30.797 | OpenAPI 형태소분석 API |
| [DistilKoBERT](https://github.com/monologg/DistilKoBERT) | SKTBrain KoBERT의 경량화 모델로 기존 12 layer에서 3 layer로 줄임 | 한국어 위키, 나무위키, 뉴스 등 10GB | | Sentencepiece |
| [KcBERT](https://github.com/Beomi/KcBERT) | 한국어 댓글 선학습 BERT 모델 | 네이버 뉴스 댓글 및 대댓글(2019.01.01 ~ 2020.06.15) 약 15.4GB (1억 1천만개 이상 문장) | 30,000 | WordPiece |
| [KR-BERT](https://github.com/snunlp/KR-BERT) | KR-BERT character, KR-BERT sub-character 모델 제공 | 2.47GB (20M 문장, 233M 단어) | KR-BERT character : 16,424 / KR-BERT sub-character : 12,367 | BidirectionalWordPiece |
| [KorPatBERT](https://github.com/kipi-ai/korpatbert) | 한국특허정보원이 배포한 특허 데이터 특화 BERT 모델, 특허문헌에서 약 666만개 주요 명사 및 복합명사 추출하여 형태소분석기 Mecab-ko 사용자 사전에 추가 후 sentencepiece를 통해 subword로 분할하는 방식 활용 (Mecab-ko Sentencepiece Patent Tokenizer) | 국내 특허문헌 약 406만건, 4억 6천장 문장, 266억 토큰 (120GB) | 21,400 | Mecab-ko Sentencepiece |
| [KB-ALBERT](https://github.com/KB-AI-Research/KB-ALBERT) | 한국어 경제 및 금융 도메인 특화 ALBERT 모델 | 일반 도메인(위키, 뉴스 등) + 금융 도메인(경제 및 금융 특화 뉴스, 리포트 등) 총 100GB (KB-ALBERT-CHAR-v2 기준) | 9,607 | 음절단위 한글 토크나이저 ( BERTWordPieceTokenizer에서 음절만 있는 형태와 비슷하며 띄어쓰기를 제외한 음절 앞에 “##” prefix 추가) |
| [KoELECTRA](https://github.com/monologg/KoELECTRA) | KoELECTRA-Base, KoELECTRA-Small 제공 | v1, v2 : 뉴스, 위키, 나무위키 등 34GB / v3 : 모두의 말뭉치 신문, 문어, 구어, 메신저, 웹 약 20GB 추가 사용 | v1, v2 : 32,200 / v3 : 35000 | WordPiece |
| [KcELECTRA](https://github.com/Beomi/KcELECTRA) | tokenizer는 huggingface의 Tokenizers 라이브러리 활용 | 뉴스 댓글 및 대댓글(2019.01.01 ~ 2021.03.09) 약 17.3GB (1억 8천만 개 이상의 문장) | 30,000 | WordPiece |
| [Dialog-KoELECTRA](https://github.com/SKplanet/Dialog-KoELECTRA) | ELECTRA기반 한국어 대화체 언어모델 | 대화체(AIHub 한국어 대화 말뭉치, 모두의말뭉치 구어 , 챗봇 데이터, KcBERT) 7GB + 문어체(모두의말뭉치 신문, 나무위키) 15GB | 40,000 | 형태소분석 기반 |


## 2️⃣ Transformer Decoder 기반

| Model | Description | Dataset | Vocab Size | Tokenizer |
|-------|-------------|---------|------------|-----------|
| [KoGPT2 (SKT-AI)](https://github.com/SKT-AI/KoGPT2) | | 한국어 위키백과, 뉴스, 모두의 말뭉치 v1.0, 청와대 국민청원 등 | 51,200 | Character BPE |
| [KoGPT (Kakaobrain)](https://github.com/kakaobrain/kogpt) | 카카오브레인에서 배포한 초거대 GPT-3 언어모델로 KoGPT6B-ryan1.5b, KoGPT6B-ryan1.5b-float16 모델 제공 | | | |



## 3️⃣ Transformer Encoder-Decoder 기반

| Model | Description | Dataset | Vocab Size | Tokenizer |
|-------|-------------|---------|------------|-----------|
| [KoBART (SKT-AI)](https://github.com/SKT-AI/KoBART) | | 한국어 위키백과, 뉴스, 모두의 말뭉치 v1.0, 청와대 국민청원 등 | 30,000 | Character BPE |
| [AsianBART](https://github.com/hyunwoongko/asian-bart) | 한국어/영어/중국어/일본어 모델 제공 | | 8,000 | |
| [T5 multilingual (Google)](https://github.com/google-research/multilingual-t5) | | mC4 코퍼스 (101개 언어) | 250,114 | Sentencepiece |
| [ET5, 한국어 이해생성 언어모델(ETRI)](https://aiopen.etri.re.kr/service_dataset.php) | ETRI 엑소브레인 연구진이 배포한 한국어 이해생성 언어모델(ET5) | Common Crawl, 위키백과, 신문기사, 방송 대본, 영화/드라마 대본, 문어/구어 등 약 136GB(12억 9천만 문장, 139억개 단어, 643억 글자) | 45,100 | Sentencepiece |
| [KE-T5 (KETI)](https://github.com/AIRC-KETI/ke-t5) | | 한국어의 경우 센터에서 확보하고 있는 비정형 코퍼스+모두의 말뭉치 일부, 영어의 경우 RealNews 데이터셋 사용 | 64,000 | Sentencepiece |
| [kolang-t5-base](https://github.com/seujung/kolang-t5-base) | | 네이버 뉴스, 위키, 모두의 말뭉치 총 20GB | 35,100 (special token 100개 포함) | Sentencepiece |



## 4️⃣ 기타

| Model | Description | Dataset | Vocab Size | Tokenizer |
|-------|-------------|---------|------------|-----------|
| [LMKor](https://github.com/kiyoungkim1/LMkor) | 한국어로 학습된 최신 언어모델 제공(BERT, ALBERT, ELECTRA, Funnel, GPT3, BERTShared) | 국내 주요 커머스 리뷰 1억개 + 블로그 형 웹사이트 2000만개 (75GB) + 모두의 말뭉치(18GB) + 위키피디아 나무위키 (6GB) | 42,000 (2,000개 unused token) | WordPiece |
| [KoBigBird](https://github.com/monologg/KoBigBird) | 최대 512개의 토큰을 다룰 수 있는 BERT의 8배인 최대 4096개의 token을 다룸 | 모두의 말뭉치, 한국어 위키, Common Crawl, 뉴스 데이터 등 | | BertTokenizer 활용 |