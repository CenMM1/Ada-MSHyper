#!/bin/bash

# Cherma https://rmiteduau-my.sharepoint.com/:u:/g/personal/s4119337_student_rmit_edu_au/IQC-30PI91rrQZSFB15E8xI6AYLHDqdd1-WBVdGayS6Wu6c?e=lQ2jd4
# mosi https://rmiteduau-my.sharepoint.com/:u:/g/personal/s4119337_student_rmit_edu_au/IQDUA10wVQu7RbqzG0YhmvabASUluTY0hmfogEoPRI0O81k?e=zNs1Vb
# mosei (25GB) https://rmiteduau-my.sharepoint.com/:u:/g/personal/s4119337_student_rmit_edu_au/IQDPgEMDUPX6QpM5NEJPwVwKAfBnzFLhwAbfdkfPBbuFhHw?e=NGoM4y
# mosi.pkl https://rmiteduau-my.sharepoint.com/:u:/g/personal/s4119337_student_rmit_edu_au/IQCZHAHn3HQ-QaS8VDgY9xkCAQKCdhgRETj6ymScUB3FEnk?e=Qm0SSO

# 下载SharePoint页面
wget -O sp.html "https://rmiteduau-my.sharepoint.com/:u:/g/personal/s4119337_student_rmit_edu_au/IQCZHAHn3HQ-QaS8VDgY9xkCAQKCdhgRETj6ymScUB3FEnk?e=Qm0SSO
"

# 从HTML中提取编码的下载URL
ENCODED_URL=$(grep -o 'https[^"]*download[^"]*tempauth[^"]*' sp.html)

# 解码\u002f为/
REAL_URL=$(echo "$ENCODED_URL" | sed 's/\\u002f/\//g')

# 下载文件，使用content-disposition获取正确文件名
wget --content-disposition -O mosi.pkl "$REAL_URL"

echo "下载完成"

# 删除临时文件
rm sp.html

# 解压文件到mosei目录
# unzip mosi.zip -d mosi

echo "解压完成"