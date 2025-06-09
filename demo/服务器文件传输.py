#!/usr/bin/env python3
import subprocess
import argparse

def upload_file(local_file, remote_user, remote_ip, remote_path):
    """上传本地文件到服务器"""
    cmd = ["scp", local_file, f"{remote_user}@{remote_ip}:{remote_path}"]
    print("执行命令：", " ".join(cmd))
    subprocess.run(cmd, check=True)

def upload_folder(local_folder, remote_user, remote_ip, remote_path):
    """上传本地文件夹到服务器"""
    cmd = ["scp", "-r", local_folder, f"{remote_user}@{remote_ip}:{remote_path}"]
    print("执行命令：", " ".join(cmd))
    subprocess.run(cmd, check=True)

def download_folder(remote_user, remote_ip, remote_path, local_destination):
    """从服务器下载文件夹到本地"""
    cmd = ["scp", "-r", f"{remote_user}@{remote_ip}:{remote_path}", local_destination]
    print("执行命令：", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="基于scp的文件传输脚本")
    parser.add_argument("--mode", default="upload_file", help=' "upload_file", "upload_folder", "download_folder" ' )
    
    parser.add_argument("--local_file", default="/home/ang/桌面/datasets/paper4/train2014.zip", help="本地文件路径（用于上传文件）")
    parser.add_argument("--local_folder", default="/home/ang/桌面/datasets/", help="本地文件夹路径（用于上传文件夹）")
    parser.add_argument("--remote_path", default="/18t/data/home/ang/datasets", help="服务器端目标路径")
    
    parser.add_argument("--remote_user", default="wangchai", help="服务器用户名，默认：gcsx")
    parser.add_argument("--remote_ip", default="192.168.110.36", help="服务器IP，默认：10.20.4.168")

    args = parser.parse_args()

    if args.mode == "upload_file":
        if not args.local_file or not args.remote_path:
            parser.error("--local_file 和 --remote_path 参数是上传文件模式必需的")
        upload_file(args.local_file, args.remote_user, args.remote_ip, args.remote_path)
    elif args.mode == "upload_folder":
        if not args.local_folder or not args.remote_path:
            parser.error("--local_folder 和 --remote_path 参数是上传文件夹模式必需的")
        upload_folder(args.local_folder, args.remote_user, args.remote_ip, args.remote_path)
    elif args.mode == "download_folder":
        if not args.remote_path or not args.local_folder:
            parser.error("--remote_path 和 --local_folder 参数是下载文件夹模式必需的")
        download_folder(args.remote_user, args.remote_ip, args.remote_path, args.local_folder)

if __name__ == "__main__":
    main()



