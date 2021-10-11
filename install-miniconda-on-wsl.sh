#!/bin/bash

# Miniconda3 installer
# WSL2のUbuntu側で行う Miniconda のセットアップスクリプト
# 最終更新: 2021/08/07

export VERSION=py37_4.10.3
export SHA256SUM=a1a7285dea0edc430b2bc7951d89bb30a2a1b32026d2a7b02aacaaa95cf69c7c

export MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-${VERSION}-Linux-x86_64.sh

# miniconda ダウンロード
# REF: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile
# Package-URL: https://docs.conda.io/en/latest/miniconda.html
wget "${MINICONDA_URL}" -O miniconda.sh -q && \
echo "${SHA256SUM} miniconda.sh" > shasum && \
if [ "${CONDA_VERSION}" != "latest" ]; then sha256sum --check --status shasum; fi && \
sudo mkdir -p /opt
# miniconda インストール
sudo bash miniconda.sh -b -p /opt/conda && \
rm miniconda.sh shasum && \
# シンボリックリンクを作成
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

#環境変数の設定
cat << 'EOS' >> ~/.profile

#Added by install-miniconda-on-wsl.sh
export PATH=/opt/conda/bin:$PATH
#Added: end

EOS

echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
echo "conda activate base" >> ~/.bashrc

# Path 設定
# REF: https://codehero.jp/anaconda/55290271/updating-anaconda-fails-environment-not-writable-error
sudo chown -R $USER:$USER /opt/conda

# 不要なパッケージをアンインストール
find /opt/conda/ -follow -type f -name '*.a' -delete && \
find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
/opt/conda/bin/conda clean -afy
