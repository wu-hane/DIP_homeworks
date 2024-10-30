FILE=cityscapes
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
mkdir -p $TARGET_DIR

# 输出下载信息到目标目录
echo "Downloading $URL dataset..." > "$TARGET_DIR/download_info.txt"

# 下载并保存到指定位置
curl -L $URL -o $TAR_FILE

# 创建目标目录（如果尚未创建）
mkdir -p $TARGET_DIR

# 解压文件
tar -zxvf $TAR_FILE -C ./datasets/

# 删除压缩文件
rm $TAR_FILE

# 获取训练和验证集中的所有jpg文件，并按名称排序后保存到相应的txt文件中
find "${TARGET_DIR}/train" -type f -name "*.jpg" | sort -V > ./train_list.txt
find "${TARGET_DIR}/val" -type f -name "*.jpg" | sort -V > ./val_list.txt