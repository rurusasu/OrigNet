#!/bin/bash
#
# create-env-on-wsl.sh
# WSL2のUbuntu上のcondaを使って仮想環境を自動作成するスクリプト
# 最終更新: 2021/08/24

# 仮想環境名(VM_NAME)とインストールするpythonのバージョン(PY_VER)を指定してください。
export VM_NAME=net
export PY_VER=3.8
# 仮想環境を一度削除
conda deactivate
conda remove -yn ${VM_NAME} --all
# 不要になったパッケージも削除
conda clean --all
# 再度仮想環境を作成
conda create -yn ${VM_NAME} python=${PY_VER}
# 仮想環境を有効にする
conda activate ${VM_NAME}

# 追加パッケージをインストールする
if [ -f ./requirements.txt ]; then
  pip3 install --upgrade pip && \
  pip3 install -r requirements.txt
fi