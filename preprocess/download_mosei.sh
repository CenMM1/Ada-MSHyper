#!/bin/bash

# 下载SharePoint页面
wget -O sp.html "https://rmiteduau-my.sharepoint.com/:x:/g/personal/s4119337_student_rmit_edu_au/IQD5GdJzvbrETLsRxD36PMwkAQyCYhGPmFHucTw-WYpd3Us?e=sAMHM9"

# 从HTML中提取编码的下载URL
ENCODED_URL=$(grep -o 'https[^"]*download[^"]*tempauth[^"]*' sp.html)

# 解码\u002f为/
REAL_URL=$(echo "$ENCODED_URL" | sed 's/\\u002f/\//g')

# 下载文件，使用content-disposition获取正确文件名
wget --content-disposition -O label.csv "$REAL_URL"

echo "下载完成"

# 删除临时文件
rm sp.html

# 解压文件到mosei目录
# unzip mosei.zip -d mosei

echo "解压完成"