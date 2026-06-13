import argparse
import os
import subprocess

try:
    from PyPDF2 import PdfMerger

    MERGE = True
except ImportError:
    print("未找到 PyPDF2。将保留未合并的 PDF 文件。")
    MERGE = False


def main(files, pdf_name):
    """
    将指定的 Jupyter Notebook 文件转换为 PDF 并合并为一个 PDF 文件。

    参数:
    files (list): Jupyter Notebook 文件列表。
    pdf_name (str): 合并后的 PDF 文件名。
    """
    # 创建临时配置文件来解决 nbconvert 7.x 的配置问题
    config_content = """c.LatexExporter.latex_count = 1
c.PDFExporter.latex_count = 1
"""
    config_file = "nbconvert_config.py"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    try:
        # 构造 nbconvert 命令的基本参数
        os_args = [
            "jupyter",
            "nbconvert",
            "--log-level",
            "CRITICAL",
            "--config",
            config_file,
            "--template",
            "chinese_template.tex.j2",
            "--to",
            "pdf",
        ]
        for f in files:
            # 依次将每个 Notebook 转换为 PDF
            os_args.append(f)
            subprocess.run(os_args)
            os_args.pop()
            print("已创建 PDF 文件 {}。".format(f))
    finally:
        # 清理临时配置文件
        if os.path.exists(config_file):
            os.remove(config_file)
    if MERGE:
        # 如果安装了 PyPDF2，则合并 PDF 文件
        pdfs = [f.split(".")[0] + ".pdf" for f in files]
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(pdf_name)
        merger.close()
        for pdf in pdfs:
            os.remove(pdf)


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser()
    # 显式传入 Notebook 文件列表，以确保生成的 PDF 文件顺序正确
    parser.add_argument("--notebooks", type=str, nargs="+", required=True)
    parser.add_argument("--pdf_filename", type=str, required=True)
    args = parser.parse_args()
    main(args.notebooks, args.pdf_filename)
