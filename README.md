# 環境設定
## WSL2 と CUDA の設定

VS Code に `Remote-Containers`，`Remote-SSH`，`Remote-WSL`，`Remote-SSH: Editing Configuration Files` の拡張機能をインストール

## WSL for Ubuntu 上で CONDA 仮想環境の構築

1. VS Code から WSL for Ubuntu にアクセス．
2. `./mnt/` 以下にあるプログラムファイルが置かれているディレクトリまで移動．
3. VS Code 上の `Tarminal` に以下のコマンドを打つ．

```bash
$ source ./create-env-on-wsl.sh
```