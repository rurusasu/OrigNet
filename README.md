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