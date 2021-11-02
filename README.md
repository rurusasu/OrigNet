# 環境設定
## WSL2 と CUDA の設定

VS Code に `Remote-Containers`，`Remote-SSH`，`Remote-WSL`，`Remote-SSH: Editing Configuration Files` の拡張機能をインストール

## WSL for Ubuntu 上で CONDA 仮想環境の構築

1. VS Code から WSL for Ubuntu にアクセス．
2. `./mnt/` 以下にあるプログラムファイルが置かれているディレクトリまで移動．
3. VS Code 上の `Tarminal` に以下のコマンドを打つ．処理中の選択肢はすべて `y` を選択．

```bash
$ source ./install-cuda-on-wsl.sh
```

```bash
$ source ./install-miniconda-on-wsl.sh
```

```bash
$ source ./create-env-on-wsl.sh
```

4. 続けて以下のコマンドで CUDA 対応の pytorch をインストールする．

```bash
$ pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## optuna-dashboard を用いた最適化の可視化
optuna-dashboard を使用することで、自動的に行った最適化のうち、もっともよかったハイパーパラメタの組み合わせを瞬時に把握することができる [optuna-dashboard](https://github.com/optuna/optuna-dashboard)．dashboard への接続は，
```cmd
$  optuna-dashboard sqlite:////${DIR_PATH}.db
```
で行える．${DIR_PATH} には，例えば `/mnt/d/My_programing/OrigNet/data/trained/1/opt_log` などの `.db` ファイルへのパスが入る．

## 共通変数
```pyhton
cfg (CfgNode): 訓練の条件設定が保存された辞書．
```