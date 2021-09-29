cd ../lib

# 事前にパッケージが存在する場合は削除する
sudo rm -rf apex
# GitHub からクローンする
git clone https://github.com/NVIDIA/apex apex
cd apex

# 仮想環境 net をアクティブに変更
conda deactivate && \
conda activate net

# apexをインストール
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./