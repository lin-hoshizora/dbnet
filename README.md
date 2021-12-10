## 新しいデータセットの準備

1. 各学習画像のテキストエリアをラベルし、VIA(Version 2)のフォーマットでJSONファイルに保存。(demoの場合、`/datasets/almex_insurance/label.json`に保存している)
2. `gen_tfrecord.py`によって、学習用tfrecordを生成
3. 必要に応じてaugmentationを予め行い、Demo用データの構造を参照して保存する。
4. データ・セットの設定ファイルを修正する。Demoの場合、`conf/train_db192mini.yaml`のdatasetsはデータセット設定ファイルのファイル名を指定している。`conf/ds_train.yaml`はトレイニング用データ・セットを定義する。`conf/ds_val.yaml`はバリデーション用データ・セットを定義する。

## ディプロイ用ツール
* tfliteモデルに変換: `tflite_cnt.py`を参照
* OpenVINOモデルに変換: `openvino_cvt.sh`を参照