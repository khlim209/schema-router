# 데이터셋 다운로드 안내

## Spider

```bash
# datasets/spider/ 에 저장
pip install gdown
gdown "https://drive.google.com/uc?export=download&id=1403EGqzIDoHMdQF4c9Bkql7PC5Ro2kb6" -O spider.zip
unzip spider.zip -d datasets/spider
```

필요 파일:
- `datasets/spider/tables.json`
- `datasets/spider/dev.json`

## Bird

```bash
# https://bird-bench.github.io/ → "Download BIRD mini-dev"
# 압축 해제 후:
# datasets/bird/dev/dev.json
# datasets/bird/dev/dev_tables.json
```

## FIBEN

DBCopilot 저자 GitHub 참고:  
https://github.com/tshu-w/DBCopilot  
(비공개일 수 있음 — 저자에게 직접 요청)

필요 파일:
- `datasets/fiben/tables.json`
- `datasets/fiben/test.json`

---

## 데이터 없이 실행 (demo 모드)

```bash
python benchmark_v2.py --datasets demo
```
